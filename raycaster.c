// =============================================================
//  Raycaster simple en Raylib 6.0
//  Inspiré de Wolfenstein 3D / technique DDA classique
// =============================================================
//
//  Compilation MSYS2 UCRT64 :
//    gcc raycaster.c -o raycaster.exe -lraylib -lm -lopengl32 -lgdi32 -lwinmm
//
//  Compilation Linux :
//    gcc raycaster.c -o raycaster -lraylib -lm
//
// =============================================================

#include "raylib.h"
#include "rlgl.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int SCREEN_W, SCREEN_H, HALF_H;
int renderW, renderH, halfRenderH;

// ----- Configuration du joueur -----
#define MOVE_SPEED   3.0f       // cases/seconde
#define ROT_SPEED    2.5f       // radians/seconde
#define FOV          (PI / 3)   // 60 degrés

// ------ Parametres de qualite/performance du rendu ------
#define RENDER_SCALE 1.5 // 1 = natif, 2 = moitié, 4 = quart
#define COL_STEP 1
#define FXAA 0
#define RAYCASTER_TYPE 1 // RAYMARCHING = 0 / DDA = 1
#define RAYMARCHING_STEP_SIZE 0.001f // Taille du pas (plus c'est petit, plus c'est précis, mais plus c'est lent)
#define RAYMARCHING_RAY_LIMIT 10.0f
#define TEX_TILE 1.0f
#define SHADOW_STEPS 8
#define PARALLAX_STEPS_MIN 64
#define PARALLAX_STEPS_MAX 180
#define PARALLAX_SCALE 0.25f
#define NUM_THREADS 8

// ----- Carte du labyrinthe -----
// 1 = mur, 0 = couloir
#define MAP_W  29
#define MAP_H  29

#define PATROL_LIGHT_SPEED 0.2f
#define CHASE_RADIUS 6.0f  // distance en cases pour déclencher la poursuite
#define NUM_LIGHTS  3
#define PATROL_LIGHTS  (NUM_LIGHTS - 1)
#define AMBIENT_LIGHT 0.03f
#define TORCHE_RADIUS 1.8f
#define TORCHE_PUISSANCE 1.5f
#define TORCHE_DISTANCE 0.5f

// Paramètres bloom des sprites light patrols
#define BLOOM_PASSES  7      // nombre de passes (1 = pas de bloom)
#define BLOOM_SPREAD  2      // écartement en pixels
#define BLOOM_ALPHA   0.3f   // intensité du halo

#define LUT_SIZE 4096
static float sinLUT[LUT_SIZE];
static float cosLUT[LUT_SIZE];

static int MAP[MAP_H][MAP_W];
// 0.0 = pas de trace, 1.0 = trace toute fraîche
static float TRACES[MAP_H][MAP_W] = {0};

int exitX, exitY;
bool afficherMap = false;
bool deadendBranchOn = true;
bool solveMaze = true;
bool torcheOn = true;
bool traceOn = false;
float pitch = 0.0f;
bool playSoundOneTime = true;
bool playBeepSound = true;
float pulseRadius = 0.0f;
float parallaxScale = PARALLAX_SCALE;

typedef enum { STATE_PLAY, STATE_WIN, STATE_LOST, STATE_LOST_BY_CHASING, STATE_MAZE_NOT_READY } GameState;
GameState gameState = STATE_PLAY;

float gameTimer  = 0.0f;   // temps écoulé en secondes
int   bestTime   = 0;      // meilleur temps en secondes (0 = pas encore de score)

bool lightReachedExit = false;

float distExit   = 0.0f;

// ----- Couleurs des murs selon la face touchée -----
// Nord/Sud plus sombres pour donner un effet de profondeur
static const Color WALL_COLOR_NS = {180, 60,  60,  255};
static const Color WALL_COLOR_EW = {220, 100, 100, 255};
static const Color FLOOR_COLOR   = { 50,  50,  50,  255};
static const Color CEIL_COLOR    = { 30,  30,  80,  255};

typedef struct { int x, y; } Dir;
static const Dir DIRS[4] = {{0, -2}, {0, 2}, {2, 0}, {-2, 0}};


typedef struct {
    float   dist;       // distance au mur (corrigée pour éviter fish-eye)
    int     side;       // 0 = mur N/S, 1 = mur E/W
    Color   color;      // couleur du mur
    float   texX;
    float   nx, ny;     // normal
    int     wallType;
    float   wx, wy;     // position exacte du point d'impact en world space
    int     mapX, mapY; // case de la grille touchée
} RayHit;

typedef struct {
    float x, y;       // position dans la carte
    float r, g, b;    // couleur
    float radius;     // portée
} Light;

typedef struct {
    Light* l;
    float dx, dy;     // Direction actuelle
    int lastGridX, lastGridY; // Pour détecter le changement de case
    bool chasing;
} MouvingLight;

Light lights[] = {
    { 3.5f, 3.5f, 1.0f, 1.0f, 0.2f, 1.5f },   // lumière patrol chaude
    { 10.f, 5.0f, 0.3f, 0.3f, 1.0f, 1.5f },   // lumière patrol froide
    { 0 }
};
int numLights = NUM_LIGHTS;

MouvingLight patrolLights[PATROL_LIGHTS] = {
    { &lights[0], 1.0f, 0.0f }, // Jaune
    { &lights[1], 1.0f, 0.0f }  // Bleue
};

typedef struct {
    Texture2D texDiffuse, texNormal, texHeight;
    Texture2D wallDataTex;
    float *wallDataBuf;
    Color *framebuffer, *framebuffersprites, *tmpFxaa;
    Texture2D screenTex;
    Color *wallPixels;
    int texW, texH;
    Color *wallNormal;
    int normW, normH;
    Color *wallHeight;
    int heightW, heightH;
    Color *floorPixels;
    int texFloorW, texFloorH;
    Color *ceilPixels;
    int texCeilW, texCeilH;
    Color *spritePixels;
    int spriteW, spriteH;
    float *zBuffer;
} Context;

// Dans les globals
typedef struct {
    float x, y;       // position monde
    float dist;       // distance au joueur (pour le tri)
    int   lightIdx;   // index dans lights[]
} Sprite;

Sprite sprites[PATROL_LIGHTS];    // une par lumière patrouille
Light tmpLightColor[NUM_LIGHTS] = {0};

typedef struct
{
    Context* ctx;

    int startX;
    int endX;

    float px;
    float py;
    float angle;

    int horizon;

} RenderThreadData;

RenderThreadData td[NUM_THREADS];
Context* gctx;

// ---------------- Prototypes de fonction ----------------------------
void ShuffleDirections(int *array, int n);
void CarveMaze(int x, int y);
void GenerateMapDFS(void);
bool SolveMaze(int startX, int startY);
void GenerateMapRandomWalk(void);
void GenerateMapWolfenstein(void);
RayHit cast_ray_dda(float px, float py, float angle, float player_angle);
RayHit cast_ray_raymarching(float px, float py, float ray_angle, float player_angle);
Sound CreateBeep(void);
void BeepDependsOnExitDistance(Sound beep, int px, int py);
void apply_fxaa(Color *fb, Color *tmp, int w, int h);
void KeysAndJoypadHandler(float* angle, float* px, float* py, float dt);
void draw_minimap(float px, float py, float angle);
typedef struct { float x, y; } Vec2;
Vec2 GetNextDirToward(int cx, int cy, int tx, int ty);
typedef struct { float x, y; } SpawnPoint;
SpawnPoint GetRandomFreeCell(float avoidX, float avoidY, float minDist);
void ResetGame(float *px, float *py);
static inline Color MixColor(Color a, Color b, float t);
void ComputeDeadEndBranches(void);
void AnimLights(float px, float py, float angle, float dt);
void GState(GameState gameState, float* px, float* py, Sound tada, Sound siren);
void RenderSprites(Context* ctx, float px, float py, float angle);
void RenderWalls(Context* ctx, int startX, int endX, float px, float py, float angle, int horizon);
void RenderFloorCeil(Context* ctx, float px, float py, float angle, int horizon);
static inline Color SampleBilinear(Color* tex, int texW, int texH,  float u, float v);
static inline float SampleHeight(Color* tex, int texW, int texH, float u, float v);
void InitLUT(void);
static inline float LUTsin(float angle);
static inline float LUTcos(float angle);
float FastInvSqrt(float x);
// --------------------------------------------------------------------

#pragma region UTILES
/*
static inline Color SampleBilinear(Color* tex, int texW, int texH, float u, float v)
{
    u -= floorf(u);
    v -= floorf(v);

    float xf = u * texW;
    float yf = v * texH;

    int x0 = (int)xf;
    int y0 = (int)yf;

    float fx = xf - x0;
    float fy = yf - y0;

    int x1 = (x0 + 1 < texW) ? x0 + 1 : 0;
    int y1 = (y0 + 1 < texH) ? y0 + 1 : 0;

    // x0/y0 ne peuvent pas dépasser texW/texH car u,v sont dans [0,1)
    // après floorf, donc xf < texW et yf < texH strictement

    Color c00 = tex[y0 * texW + x0];
    Color c10 = tex[y0 * texW + x1];
    Color c01 = tex[y1 * texW + x0];
    Color c11 = tex[y1 * texW + x1];

    float r = c00.r*(1-fx)*(1-fy) + c10.r*fx*(1-fy) + c01.r*(1-fx)*fy + c11.r*fx*fy;
    float g = c00.g*(1-fx)*(1-fy) + c10.g*fx*(1-fy) + c01.g*(1-fx)*fy + c11.g*fx*fy;
    float b = c00.b*(1-fx)*(1-fy) + c10.b*fx*(1-fy) + c01.b*(1-fx)*fy + c11.b*fx*fy;

    return (Color){ (unsigned char)r, (unsigned char)g, (unsigned char)b, 255 };
}
*/

static inline Color SampleBilinear(Color* tex, int texW, int texH, float u, float v)
{
    u -= floorf(u);
    v -= floorf(v);

    int x0 = (int)(u * texW);          // évite le *( texW-1) + remapping
    int y0 = (int)(v * texH);
    float fx = u * texW - x0;
    float fy = v * texH - y0;

    // Clamp avec masque (branchless)
    int x1 = x0 + 1; if (x1 >= texW) x1 = 0;   // wrap plutôt que clamp → meilleure cohérence avec fmodf
    int y1 = y0 + 1; if (y1 >= texH) y1 = 0;

    Color c00 = tex[y0 * texW + x0];
    Color c10 = tex[y0 * texW + x1];
    Color c01 = tex[y1 * texW + x0];
    Color c11 = tex[y1 * texW + x1];

    //int ifx = (int)(fx * 256);         // virgule fixe Q8 → évite 6 multiplications float
    //int ify = (int)(fy * 256);
    //int ifxi = 256 - ifx;
    //int ifyi = 256 - ify;

    int ifx  = (int)(fx * 256); if (ifx  > 255) ifx  = 255;
    int ify  = (int)(fy * 256); if (ify  > 255) ify  = 255;
    int ifxi = 256 - ifx;
    int ifyi = 256 - ify;

    Color out;
    out.r = (unsigned char)(((c00.r*ifxi + c10.r*ifx)*ifyi + (c01.r*ifxi + c11.r*ifx)*ify) >> 16);
    out.g = (unsigned char)(((c00.g*ifxi + c10.g*ifx)*ifyi + (c01.g*ifxi + c11.g*ifx)*ify) >> 16);
    out.b = (unsigned char)(((c00.b*ifxi + c10.b*ifx)*ifyi + (c01.b*ifxi + c11.b*ifx)*ify) >> 16);
    out.a = 255;
    return out;
}

float FastInvSqrt(float x) {
    float xhalf = 0.5f * x;
    int i;
    // Utilisation de memcpy pour éviter les violations de typage strict (strict-aliasing)
    memcpy(&i, &x, sizeof(i));
    i = 0x5f3759df - (i >> 1); // Le fameux nombre magique
    memcpy(&x, &i, sizeof(x));
    x = x * (1.5f - xhalf * x * x); // 1ère itération de Newton (suffisante pour de la 3D)
    // x = x * (1.5f - xhalf * x * x); // Active cette 2e itération si tu as besoin de plus de précision
    return x;
}

void InitLUT(void)
{
    for (int i = 0; i < LUT_SIZE; i++)
    {
        float angle = (float)i * (2.0f * PI / LUT_SIZE);
        sinLUT[i] = sinf(angle);
        cosLUT[i] = cosf(angle);
    }
}

static inline float LUTsin(float angle)
{
    // Normalise l'angle dans [0, 2PI]
    angle = fmodf(angle, 2.0f * PI);
    if (angle < 0) angle += 2.0f * PI;
    int idx = (int)(angle * LUT_SIZE / (2.0f * PI)) & (LUT_SIZE - 1);
    return sinLUT[idx];
}

static inline float LUTcos(float angle)
{
    angle = fmodf(angle, 2.0f * PI);
    if (angle < 0) angle += 2.0f * PI;
    int idx = (int)(angle * LUT_SIZE / (2.0f * PI)) & (LUT_SIZE - 1);
    return cosLUT[idx];
}


static inline Color MixColor(Color a, Color b, float t)
{
    return (Color){
        (unsigned char)(a.r + (b.r - a.r) * t),
        (unsigned char)(a.g + (b.g - a.g) * t),
        (unsigned char)(a.b + (b.b - a.b) * t),
        255
    };
}

void ResetGame(float *px, float *py)
{
    GenerateMapRandomWalk();
    ComputeDeadEndBranches();

    // Spawner le joueur loin de la sortie
    SpawnPoint sp = GetRandomFreeCell(exitX, exitY, 8.0f);
    *px = sp.x;
    *py = sp.y;

    // Spawner les lumières adverses
    float lastX = *px, lastY = *py;
    for (int i = 0; i < PATROL_LIGHTS; i++)
    {
        SpawnPoint lp = GetRandomFreeCell(lastX, lastY, 5.0f);
        lights[i].x = lp.x;
        lights[i].y = lp.y;
        patrolLights[i].lastGridX = (int)lp.x;
        patrolLights[i].lastGridY = (int)lp.y;
        Vec2 d = GetNextDirToward((int)lp.x, (int)lp.y, exitX, exitY);
        patrolLights[i].dx = (d.x != 0 || d.y != 0) ? d.x : 1.0f;
        patrolLights[i].dy = (d.x != 0 || d.y != 0) ? d.y : 0.0f;

        // Chaque lumière spawne loin de la précédente
        lastX = lp.x;
        lastY = lp.y;
    }

    pitch     = 0.0f;
    gameTimer = 0.0f;
    parallaxScale = PARALLAX_SCALE;
    gameState = STATE_PLAY;
    traceOn   = false;
    playSoundOneTime = true;
    playBeepSound    = true;

    for (int i=0; i<PATROL_LIGHTS; i++)
    {
        patrolLights[i].chasing = false;

        tmpLightColor[i].r = patrolLights[i].l->r;
        tmpLightColor[i].g = patrolLights[i].l->g;
        tmpLightColor[i].b = patrolLights[i].l->b;
    }

    if (!SolveMaze(*px, *py)) gameState = STATE_MAZE_NOT_READY;
    memset(TRACES, 0, sizeof(TRACES));
}

// =============================================================
//  Minimap (vue du dessus pour déboguer)
// =============================================================

void draw_minimap(float px, float py, float angle)
{
    const int TILE  = 8;    // taille d'une case en pixels
    const int OX    = 10;   // offset en haut à gauche
    const int OY    = 10;

    // Grille
    for (int y = 0; y < MAP_H; y++) {
        for (int x = 0; x < MAP_W; x++) {
            Color c = MAP[y][x] ? (Color){200,200,200,200}
                                : (Color){ 20, 20, 20,180};
            DrawRectangle(OX + x*TILE, OY + y*TILE, TILE-1, TILE-1, c);
            
            // Dessin de la trace par-dessus le couloir
            if (MAP[y][x] == 0 && TRACES[y][x] > 0) {
                Color traceColor = { 255, 255, 255, (unsigned char)(TRACES[y][x] * 100) }; 
                DrawRectangle(OX + x*TILE, OY + y*TILE, TILE-1, TILE-1, traceColor);
            }

            if (MAP[y][x]==2) DrawRectangle(OX + x*TILE, OY + y*TILE, TILE-1, TILE-1, GOLD);
        }
    }

    // Joueur
    int ppx = OX + (int)(px * TILE);
    int ppy = OY + (int)(py * TILE);
    DrawCircle(ppx, ppy, 3, YELLOW);

    // Direction
    DrawLine(ppx, ppy,
             ppx + (int)(cosf(angle) * 10),
             ppy + (int)(sinf(angle) * 10),
             YELLOW);

    // Lumières patrouilles
    for (int i = 0; i < PATROL_LIGHTS; i++) {
        int lpx = OX + (int)(patrolLights[i].l->x * TILE);
        int lpy = OY + (int)(patrolLights[i].l->y * TILE);
        //Color lc = (i == 0) ? (Color){255, 200, 50, 255} : (Color){50, 150, 255, 255};
        Color lc = (Color){(int)(fminf(patrolLights[i].l->r, 1.0f) * 255.0), 
                           (int)(fminf(patrolLights[i].l->g, 1.0f) * 255.0), 
                           (int)(fminf(patrolLights[i].l->b, 1.0f) * 255.0), 255};
        DrawCircle(lpx, lpy, 4, lc);

        // Direction de la lumière
        DrawLine(lpx, lpy,
                lpx + (int)(patrolLights[i].dx * 8),
                lpy + (int)(patrolLights[i].dy * 8),
                lc);
    }
}

void apply_fxaa(Color *fb, Color *tmp, int w, int h)
{
    memcpy(tmp, fb, w * h * sizeof(Color));

    for (int y = 1; y < h - 1; y++)
    for (int x = 1; x < w - 1; x++)
    {
        Color c  = tmp[y*w + x];
        Color n  = tmp[(y-1)*w + x];
        Color s  = tmp[(y+1)*w + x];
        Color e  = tmp[y*w + x+1];
        Color ww = tmp[y*w + x-1];

        // Détection de contour (luminance)
        float lC = 0.299f*c.r  + 0.587f*c.g  + 0.114f*c.b;
        float lN = 0.299f*n.r  + 0.587f*n.g  + 0.114f*n.b;
        float lS = 0.299f*s.r  + 0.587f*s.g  + 0.114f*s.b;
        float lE = 0.299f*e.r  + 0.587f*e.g  + 0.114f*e.b;
        float lW = 0.299f*ww.r + 0.587f*ww.g + 0.114f*ww.b;

        float edge = fabsf(lN-lS) + fabsf(lE-lW);

        if (edge > 8.0f)  // seuil — ajuster selon le rendu
        {
            // Moyenne des voisins sur les contours
            fb[y*w + x] = (Color){
                (c.r + n.r + s.r + e.r + ww.r) / 5,
                (c.g + n.g + s.g + e.g + ww.g) / 5,
                (c.b + n.b + s.b + e.b + ww.b) / 5,
                255
            };
        }
    }
}
#pragma endregion


#pragma region GENERATION MAZE MAP
static int DEADEND_BRANCH[MAP_H][MAP_W] = {0};

void ComputeDeadEndBranches(void)
{
    memset(DEADEND_BRANCH, 0, sizeof(DEADEND_BRANCH));

    int tempMap[MAP_H][MAP_W];
    memcpy(tempMap, MAP, sizeof(MAP));

    // depth[y][x] = longueur de la branche depuis le cul-de-sac
    int depth[MAP_H][MAP_W];
    memset(depth, 0, sizeof(depth));

    #define MAX_BRANCH_DEPTH 3  // ajuster selon le goût

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int y = 1; y < MAP_H - 1; y++)
        for (int x = 1; x < MAP_W - 1; x++)
        {
            if (tempMap[y][x] != 0) continue;
            if (x == exitX && y == exitY) continue;

            int free = 0;
            if (tempMap[y][x-1] == 0) free++;
            if (tempMap[y][x+1] == 0) free++;
            if (tempMap[y-1][x] == 0) free++;
            if (tempMap[y+1][x] == 0) free++;

            if (free == 1 && depth[y][x] < MAX_BRANCH_DEPTH)
            {
                DEADEND_BRANCH[y][x] = 1;
                tempMap[y][x] = -1;

                // Propager la profondeur au voisin libre
                int ddx[] = {-1,1,0,0};
                int ddy[] = {0,0,-1,1};
                for (int d = 0; d < 4; d++) {
                    int nx = x + ddx[d];
                    int ny = y + ddy[d];
                    if (nx < 0 || ny < 0 || nx >= MAP_W || ny >= MAP_H) continue;
                    if (tempMap[ny][nx] != 0) continue;
                    // Le voisin hérite de la profondeur +1
                    if (depth[ny][nx] < depth[y][x] + 1)
                        depth[ny][nx] = depth[y][x] + 1;
                }

                changed = true;
            }
        }
    }
}

// Trouve une case libre aléatoire dans la map
// Evite la sortie et une distance minimale d'un autre point
SpawnPoint GetRandomFreeCell(float avoidX, float avoidY, float minDist)
{
    int attempts = 0;
    while (attempts < 1000)
    {
        int rx = GetRandomValue(1, MAP_W - 2);
        int ry = GetRandomValue(1, MAP_H - 2);

        if (MAP[ry][rx] != 0) { attempts++; continue; }

        float dx = rx - avoidX;
        float dy = ry - avoidY;
        if (sqrtf(dx*dx + dy*dy) < minDist) { attempts++; continue; }

        return (SpawnPoint){ rx + 0.5f, ry + 0.5f };
    }
    // Fallback si rien trouvé
    return (SpawnPoint){ 1.5f, 1.5f };
}

// Mélange un tableau de directions (Fisher-Yates simple)
void ShuffleDirections(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = GetRandomValue(0, i);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void CarveMaze(int x, int y) {
    int order[] = {0, 1, 2, 3};
    ShuffleDirections(order, 4);

    for (int i = 0; i < 4; i++) {
        int dx = DIRS[order[i]].x;
        int dy = DIRS[order[i]].y;
        int nx = x + dx;
        int ny = y + dy;

        // Si la case cible est dans les limites et est encore un mur
        if (nx > 0 && nx < MAP_W - 1 && ny > 0 && ny < MAP_H - 1) {
            if (MAP[ny][nx] == 1) {
                // On casse le mur entre la case actuelle et la suivante
                MAP[y + dy / 2][x + dx / 2] = 0;
                // On casse la case suivante
                MAP[ny][nx] = 0;
                // On continue la récursion
                CarveMaze(nx, ny);
            }
        }
    }
}

void GenerateMapDFS(void) {
    // 1. Initialiser tout en murs
    for (int y = 0; y < MAP_H; y++) {
        for (int x = 0; x < MAP_W; x++) {
            MAP[y][x] = 1;
        }
    }

    // 2. Choisir un point de départ impair (pour l'alignement de la grille)
    int startX = 1;
    int startY = 1;
    MAP[startY][startX] = 0;

    // 3. Lancer l'algorithme
    CarveMaze(startX, startY);

    // 3. Sortie de la map
    int endX = exitX = MAP_W - 1;
    int endy = exitY = GetRandomValue(1, MAP_W - 2);
    MAP[endy][endX] = 2;
}

typedef struct { int x, y; } Coord;

// Pour reconstruire le chemin : on stocke le parent de chaque case
Coord parent[MAP_H][MAP_W];

bool SolveMaze(int startX, int startY) {
    // 1. Initialisation
    bool visited[MAP_H][MAP_W] = {false};
    for(int y=0; y<MAP_H; y++) 
        for(int x=0; x<MAP_W; x++) parent[y][x] = (Coord){-1, -1};

    // File pour le BFS (simple tableau utilisé comme file)
    Coord queue[MAP_W * MAP_H];
    int head = 0, tail = 0;

    // Ajouter le départ
    queue[tail++] = (Coord){startX, startY};
    visited[startY][startX] = true;

    Coord target = {-1, -1};

    // 2. Exploration
    while (head < tail) {
        Coord curr = queue[head++];

        // Si on a trouvé la sortie (case 2)
        if (MAP[curr.y][curr.x] == 2) {
            target = curr;
            break;
        }

        // Tester les 4 directions
        Coord dirs[4] = {{0,1}, {0,-1}, {1,0}, {-1,0}};
        for (int i = 0; i < 4; i++) {
            int nx = curr.x + dirs[i].x;
            int ny = curr.y + dirs[i].y;

            if (nx >= 0 && nx < MAP_W && ny >= 0 && ny < MAP_H &&
                MAP[ny][nx] != 1 && !visited[ny][nx]) {
                visited[ny][nx] = true;
                parent[ny][nx] = curr;
                queue[tail++] = (Coord){nx, ny};
            }
        }
    }

    // 3. Remonter le fil d'Ariane et remplir TRACES
    // On remet TRACES à zéro d'abord
    for(int y=0; y<MAP_H; y++) for(int x=0; x<MAP_W; x++) TRACES[y][x] = 0;

    if (target.x != -1) {
        Coord pathStep = target;
        while (pathStep.x != -1) {
            TRACES[pathStep.y][pathStep.x] = 1.0f; // On marque la solution
            pathStep = parent[pathStep.y][pathStep.x];
        }
    }

    return (target.x != -1);
}


void GenerateMapRandomWalk(void)
{
    // 1. Tout en murs
    for (int y = 0; y < MAP_H; y++)
        for (int x = 0; x < MAP_W; x++)
            MAP[y][x] = 1;

    bool visited[MAP_H][MAP_W] = {false};

    // Compter les cases "creusables" (positions impaires)
    int total = ((MAP_W - 1) / 2) * ((MAP_H - 1) / 2);
    int visitedCount = 0;

    // Départ aléatoire sur une position impaire
    int cx = 1, cy = 1;
    MAP[cy][cx] = 0;
    visited[cy][cx] = true;
    visitedCount++;

    // Directions possibles (pas de 2 cases)
    const int dx[] = { 0,  0, 2, -2 };
    const int dy[] = { 2, -2, 0,  0 };

    while (visitedCount < total)
    {
        // Choisir une direction aléatoire
        int dir = GetRandomValue(0, 3);
        int nx  = cx + dx[dir];
        int ny  = cy + dy[dir];

        // Rester dans les limites (positions impaires)
        if (nx < 1 || nx >= MAP_W - 1 || ny < 1 || ny >= MAP_H - 1)
            continue;

        if (!visited[ny][nx])
        {
            // Casser le mur entre (cx,cy) et (nx,ny)
            MAP[cy + dy[dir]/2][cx + dx[dir]/2] = 0;
            MAP[ny][nx] = 0;
            visited[ny][nx] = true;
            visitedCount++;
        }

        cx = nx;
        cy = ny;
    }

    // Sortie
    int endX = exitX = MAP_W - 1;
    int endY = exitY = GetRandomValue(1, MAP_H - 2);
    MAP[endY][endX] = 2;
}

void GenerateMapWolfenstein(void)
{
    // 1. On remplit tout de vide (0)
    for (int y = 0; y < MAP_H; y++)
        for (int x = 0; x < MAP_W; x++)
            MAP[y][x] = 0;

    // 2. Bordures extérieures obligatoires
    for (int x = 0; x < MAP_W; x++) { MAP[0][x] = 1; MAP[MAP_H-1][x] = 1; }
    for (int y = 0; y < MAP_H; y++) { MAP[y][0] = 1; MAP[y][MAP_W-1] = 1; }

    // 3. Création de "piliers" ou blocs isolés (Tracé typique Wolf3D)
    for (int y = 2; y < MAP_H - 2; y += 4) {
        for (int x = 2; x < MAP_W - 2; x += 4) {
            // Une chance sur deux de créer un bloc de 2x2 murs
            if (GetRandomValue(0, 10) > 4) {
                MAP[y][x] = 1; MAP[y+1][x] = 1;
                MAP[y][x+1] = 1; MAP[y+1][x+1] = 1;
            }
        }
    }

    // 4. Ajout de grandes salles rectangulaires (Zones de combat)
    int numRooms = 4;
    for (int r = 0; r < numRooms; r++) {
        int rw = GetRandomValue(4, 8);
        int rh = GetRandomValue(4, 8);
        int rx = GetRandomValue(1, MAP_W - rw - 1);
        int ry = GetRandomValue(1, MAP_H - rh - 1);

        for (int y = ry; y < ry + rh; y++) {
            for (int x = rx; x < rx + rw; x++) {
                // On vide l'intérieur de la salle
                MAP[y][x] = 0;
                // On peut ajouter un pilier central décoratif dans la salle
                if (rw > 5 && rh > 5 && x == rx + rw/2 && y == ry + rh/2) 
                    MAP[y][x] = 1;
            }
        }
    }

    // 5. Garantir un point de départ dégagé
    MAP[1][1] = 0; MAP[1][2] = 0; MAP[2][1] = 0;

    // 6. Placement de la sortie (Case 2) loin du joueur
    exitX = MAP_W - 2;
    exitY = MAP_H - 2;
    MAP[exitY][exitX] = 2;
}

Vec2 GetNextDirToward(int cx, int cy, int tx, int ty)
{
    // BFS depuis position courante vers cible
    int prev[MAP_H][MAP_W][2];  // stocke le parent de chaque case
    memset(prev, -1, sizeof(prev));

    typedef struct { int x, y; } Cell;
    Cell queue[MAP_W * MAP_H];
    int head = 0, tail = 0;

    queue[tail++] = (Cell){cx, cy};
    prev[cy][cx][0] = cx;
    prev[cy][cx][1] = cy;

    bool found = false;
    while (head < tail && !found) {
        Cell curr = queue[head++];
        if (curr.x == tx && curr.y == ty) { found = true; break; }

        int ddx[] = {1,-1,0,0};
        int ddy[] = {0,0,1,-1};
        for (int d = 0; d < 4; d++) {
            int nx = curr.x + ddx[d];
            int ny = curr.y + ddy[d];
            if (nx < 0 || ny < 0 || nx >= MAP_W || ny >= MAP_H) continue;
            if (MAP[ny][nx] == 1) continue;
            if (prev[ny][nx][0] != -1) continue;
            prev[ny][nx][0] = curr.x;
            prev[ny][nx][1] = curr.y;
            queue[tail++] = (Cell){nx, ny};
        }
    }

    if (!found) return (Vec2){0, 0};

    // Remonter le chemin jusqu'au premier pas
    Cell step = {tx, ty};
    while (prev[step.y][step.x][0] != cx || prev[step.y][step.x][1] != cy) {
        int px2 = prev[step.y][step.x][0];
        int py2 = prev[step.y][step.x][1];
        if (px2 == cx && py2 == cy) break;
        step = (Cell){px2, py2};
    }

    return (Vec2){step.x - cx, step.y - cy};
}
#pragma endregion


#pragma region DDA ET RAYMARCHING
// =============================================================
//  Algorithme DDA (Digital Differential Analyzer)
//
//  Principe : on lance un rayon depuis la position du joueur
//  dans une direction donnée. On avance case par case dans la
//  grille en choisissant toujours la frontière (verticale ou
//  horizontale) la plus proche.
// =============================================================

RayHit cast_ray_dda(float px, float py, float angle, float player_angle)
{
    // Direction du rayon
    // Le DDA a besoin d'un vecteur de longueur 1 pour que delta_x = 1/rdx soit correct.
    // cosf(angle) et sinf(angle) font exactement ça — ils donnent directement un vecteur normalisé depuis un angle.
    //
    // angle → (cosf(angle), sinf(angle))  longueur toujours = 1
    // point → normalise(target - origin)  longueur ramenée à 1
    float rdx = LUTcos(angle);
    float rdy = LUTsin(angle);

    // Position du player dans la map
    int mx = (int)px;
    int my = (int)py;

    // les cases font toujours 1×1 dans ta grille.
    // delta_x = distance parcourue par le rayon pour traverser 1 unité en X
    // delta_y = distance parcourue par le rayon pour traverser 1 unité en Y
    // rayon diagonal (45°)        rayon presque horizontal
    // rdx = 0.707, rdy = 0.707    rdx = 0.99, rdy = 0.14
    //
    // delta_x = 1/0.707 = 1.41    delta_x = 1/0.99 = 1.01  (presque 1)
    // delta_y = 1/0.707 = 1.41    delta_y = 1/0.14 = 7.14  (très long)
    //
    // Le rayon est presque horizontal donc :
    //
    // il croise une frontière verticale (en X) tous les 1.01 — souvent
    // il croise une frontière horizontale (en Y) tous les 7.14 — rarement
    //
    // La boucle DDA choisira donc side_x presque à chaque itération, et n'avancera en my qu'une fois toutes les ~7 cases.
    float delta_x = (rdx == 0) ? 1e30f : fabsf(1.0f / rdx);
    float delta_y = (rdy == 0) ? 1e30f : fabsf(1.0f / rdy);

    int step_x = (rdx < 0) ? -1 : +1;
    int step_y = (rdy < 0) ? -1 : +1;

    // Direction Formule         Signification
    // rdx > 0   mx + 1 - px     distance jusqu'au bord droit de la case
    // rdx < 0   px - mx         distance jusqu'au bord gauche de la case
    // rdy > 0   my + 1 - py     distance jusqu'au bord bas
    // rdy < 0   py - my         distance jusqu'au bord haut
    // Multiplié par delta_x/delta_y pour convertir cette fraction de case 
    // en distance réelle parcourue par le rayon.
    float side_x = (rdx < 0) ? (px - mx) * delta_x : (mx + 1.0f - px) * delta_x;
    float side_y = (rdy < 0) ? (py - my) * delta_y : (my + 1.0f - py) * delta_y;

    int side = 0;

    // Boucle DDA — ne pas toucher
    while (1) {
        if (side_x < side_y) { side_x += delta_x; mx += step_x; side = 0; }
        else                 { side_y += delta_y; my += step_y; side = 1; }

        if (mx < 0 || my < 0 || mx >= MAP_W || my >= MAP_H) break;
        if (MAP[my][mx] != 0) break;
    }

    float perp_dist = (side == 0) ? side_x - delta_x : side_y - delta_y;
    if (perp_dist < 0.001f) perp_dist = 0.001f;    

    // texX
    float hitX = (side == 0) ? py + perp_dist * rdy : px + perp_dist * rdx;
    hitX -= floorf(hitX);

    RayHit hit;
    hit.dist  = perp_dist * LUTcos(angle - player_angle);
    hit.side  = side;
    hit.color = (side == 0) ? WALL_COLOR_NS : WALL_COLOR_EW;
    hit.wallType = MAP[my][mx];

    int traceX = (side == 0) ? mx - step_x : mx;
    int traceY = (side == 1) ? my - step_y : my;
    if (traceX >= 0 && traceX < MAP_W && traceY >= 0 && traceY < MAP_H)
    {
        if (TRACES[traceY][traceX] == 1.0f) hit.wallType = 3;
        if (TRACES[traceY][traceX] == 2.0f) hit.wallType = 4;
        if (TRACES[traceY][traceX] == 3.0f) hit.wallType = 5;
    }

    if (MAP[my][mx] == 2) hit.wallType = 2;

    hit.texX  = fmodf(hitX * TEX_TILE, 1.0f);

    float euc = hit.dist / LUTcos(angle - player_angle);

    hit.mapX = mx;
    hit.mapY = my;

    // Normale
    if (side == 0) { hit.nx = (float)-step_x; hit.ny = 0.0f; }
    else           { hit.nx = 0.0f;           hit.ny = (float)-step_y; }

    // Position exacte du point d'impact
    if (hit.side == 0)
    {
        hit.wx = (float)mx + (step_x < 0 ? 1.0f : 0.0f);
        hit.wy = py + euc * rdy;
    }
    else
    {
        hit.wx = px + euc * rdx;
        hit.wy = (float)my + (step_y < 0 ? 1.0f : 0.0f);
    }   

    return hit;
}


// =============================================================
//  Algorithme Raymraching par step constant
//
//  Principe : on lance un rayon depuis la position du joueur
//  dans une direction donnée. On avance case par case dans la
//  grille en choisissant toujours la frontière (verticale ou
//  horizontale) la plus proche.
// =============================================================

RayHit cast_ray_raymarching(float px, float py, float ray_angle, float player_angle)
{
    // Direction du rayon
    float rdx = LUTcos(ray_angle);
    float rdy = LUTsin(ray_angle);

    // Taille du pas (plus c'est petit, plus c'est précis, mais plus c'est lent)
    const float STEP_SIZE = RAYMARCHING_STEP_SIZE; 
    float distance = 0.0f;
    
    int mx, my;

    int prev_x, prev_y;

    // On avance petit à petit
    while (distance < RAYMARCHING_RAY_LIMIT) // Limite de portée du rayon
    {
        distance += STEP_SIZE;
        
        // Position actuelle du bout du rayon
        float cur_x = px + rdx * distance;
        float cur_y = py + rdy * distance;

        // Conversion en coordonnées de grille
        mx = (int)cur_x;
        my = (int)cur_y;

        // Sortie de carte ?
        if (mx < 0 || my < 0 || mx >= MAP_W || my >= MAP_H) break;

        prev_x = (int)(cur_x - rdx * STEP_SIZE);
        prev_y = (int)(cur_y - rdy * STEP_SIZE);

        // Collision murale ?
        if (MAP[my][mx] != 0) {            
            break;
        }
    }

    // Correction Fish-eye (nécessaire même ici)
    // On multiplie par le cosinus de la différence d'angle
    // Note : il faudra passer l'angle du joueur en paramètre pour être parfait
    
    RayHit hit;
    hit.dist  = distance * LUTcos(ray_angle - player_angle); 
    hit.wallType = MAP[my][mx];    

    if (TRACES[prev_y][prev_x] == 1.0f) hit.wallType = 3;
    if (TRACES[prev_y][prev_x] == 2.0f) hit.wallType = 4;
    if (TRACES[prev_y][prev_x] == 3.0f) hit.wallType = 5;

    if (MAP[my][mx] == 2) hit.wallType = 2;

    hit.mapX = mx;
    hit.mapY = my;

    hit.wx = px + rdx * distance;
    hit.wy = py + rdy * distance;

    float localX = hit.wx - floorf(hit.wx);
    float localY = hit.wy - floorf(hit.wy);

    float dLeft   = localX;
    float dRight  = 1.0f - localX;
    float dTop    = localY;
    float dBottom = 1.0f - localY;

    float minDist = dLeft;
    hit.nx=-1; hit.ny=0;

    if(dRight < minDist){
        minDist=dRight;
        hit.nx=1; hit.ny=0;
    }
    if(dTop < minDist){
        minDist=dTop;
        hit.nx=0; hit.ny=-1;
    }
    if(dBottom < minDist){
        hit.nx=0; hit.ny=1;
    }
    
    float hitX;

    if (fabsf(hit.nx) > 0.5f)
    {
        // mur vertical → utiliser Y
        hitX = hit.wy;
    }
    else
    {
        // mur horizontal → utiliser X
        hitX = hit.wx;
    }

    hitX -= floorf(hitX);
    hit.texX = fmodf(hitX * TEX_TILE, 1.0f);

    hit.side = (fabsf(hit.nx) > 0.5f) ? 0 : 1;

    return hit;
}
#pragma endregion


#pragma region SONS
// =============================================================
//  Creation de sons proceduraux
// =============================================================

Sound CreateTada(void)
{
    int sampleRate = 44100;

    // Tempo rapide : noire = 0.15s, blanche = 0.30s
    float noireLen  = 0.15f;
    float blancheLen = 0.30f;
    int totalSamples = (int)((noireLen + blancheLen) * sampleRate);

    float *samples = malloc(totalSamples * sizeof(float));
    memset(samples, 0, totalSamples * sizeof(float));

    // La4 = 440 Hz — modérément aigu
    float freq = 440.0f;

    int noireSamples   = (int)(noireLen   * sampleRate);
    int blancheSamples = (int)(blancheLen * sampleRate);

    // Note 1 : noire
    for (int i = 0; i < noireSamples; i++)
    {
        float t   = (float)i / sampleRate;
        float env = 1.0f;
        if (i < 200)                          env = (float)i / 200.0f;           // attaque
        if (i > noireSamples - 800)           env = (float)(noireSamples - i) / 800.0f; // release

        samples[i] = env * (
            0.6f * sinf(2.0f * PI * freq * t) +
            0.3f * sinf(2.0f * PI * freq * 2.0f * t) +
            0.1f * sinf(2.0f * PI * freq * 3.0f * t)
        );
    }

    // Note 2 : blanche (même hauteur)
    for (int i = 0; i < blancheSamples; i++)
    {
        float t   = (float)i / sampleRate;
        float env = 1.0f;
        if (i < 200)                            env = (float)i / 200.0f;
        if (i > blancheSamples - 1500)          env = (float)(blancheSamples - i) / 1500.0f;

        samples[noireSamples + i] = env * (
            0.6f * sinf(2.0f * PI * freq * t) +
            0.3f * sinf(2.0f * PI * freq * 2.0f * t) +
            0.1f * sinf(2.0f * PI * freq * 3.0f * t)
        );
    }

    Wave wave = {
        .frameCount = totalSamples,
        .sampleRate = sampleRate,
        .sampleSize = 32,
        .channels   = 1,
        .data       = samples
    };
    Sound sound = LoadSoundFromWave(wave);
    free(samples);
    return sound;
}

Sound CreateSiren(void)
{
    int sampleRate  = 44100;
    float duration  = 4.0f;
    int sampleCount = (int)(sampleRate * duration);
    float *samples  = malloc(sampleCount * sizeof(float));

    // --- Paramètres du Pitch Global (Le Decay) ---
    float baseFreqStart = 880.0f; // Commence haut (La4)
    float baseFreqEnd   = 220.0f; // Finit bas (La2)
    float amplitude     = 150.0f; // Oscillation autour de la base (+/- 150Hz)

    // --- Paramètres du Rythme (LFO) ---
    float modFreqStart  = 5.0f;   // 5 "cycles" par seconde (plus rapide)
    float modFreqEnd    = 1.0f;   // Ralentit jusqu'à 1 cycle par seconde

    float phaseSound = 0.0f;
    float phaseMod   = 0.0f;

    for (int i = 0; i < sampleCount; i++)
    {
        float prog = (float)i / sampleCount;

        // 1. Le Decay Global : La note centrale descend
        float currentBaseFreq = baseFreqStart + (baseFreqEnd - baseFreqStart) * prog;

        // 2. Le LFO : La vitesse de l'oscillation ralentit
        float currentModFreq = modFreqStart + (modFreqEnd - modFreqStart) * prog;
        
        phaseMod += (2.0f * PI * currentModFreq) / (float)sampleRate;
        if (phaseMod > 2.0f * PI) phaseMod -= 2.0f * PI;

        // 3. Calcul du Pitch Final 
        // Fréquence de base qui descend + l'oscillation sinusoïdale
        float currentPitch = currentBaseFreq + (sinf(phaseMod) * amplitude);

        // Sécurité : on évite les fréquences négatives ou trop basses
        if (currentPitch < 20.0f) currentPitch = 20.0f;

        // 4. Accumulation de la phase sonore
        phaseSound += (2.0f * PI * currentPitch) / (float)sampleRate;
        if (phaseSound > 2.0f * PI) phaseSound -= 2.0f * PI;

        // 5. Enveloppe de volume (Fade out final pour la propreté)
        float volumeEnv = 1.0f;
        if (i > sampleCount - 4000) 
            volumeEnv = (float)(sampleCount - i) / 4000.0f;

        // Rendu : Signal riche en harmoniques (type onde de scie/carrée adoucie)
        samples[i] = volumeEnv * (
            0.5f * sinf(phaseSound) + 
            0.2f * sinf(phaseSound * 2.0f) + 
            0.1f * sinf(phaseSound * 3.0f)
        );
    }

    Wave wave = {
        .frameCount = sampleCount,
        .sampleRate = sampleRate,
        .sampleSize = 32,
        .channels   = 1,
        .data       = samples
    };
    
    Sound sound = LoadSoundFromWave(wave);
    free(samples);
    return sound;
}

Sound CreateBeep(void) {
    int sampleRate = 44100;
    int sampleCount = sampleRate * 0.1; // Durée du bip : 0.1 seconde
    float *samples = (float *)malloc(sampleCount * sizeof(float));

    for (int i = 0; i < sampleCount; i++) {
        // Génère une onde sinusoïdale simple à 440Hz (La)
        samples[i] = sinf(2.0f * PI * 440.0f * i / sampleRate);
        
        // Optionnel : Petit "fade out" pour éviter le "clic" sec à la fin
        if (i > sampleCount - 100) {
            samples[i] *= (float)(sampleCount - i) / 100.0f;
        }
    }

    Wave wave = {
        .frameCount = sampleCount,
        .sampleRate = sampleRate,
        .sampleSize = 32,      // 32 bits (float)
        .channels = 1,         // Mono
        .data = samples
    };

    Sound sound = LoadSoundFromWave(wave);
    free(samples); // On libère la mémoire temporaire, le son est maintenant dans le buffer OpenAL/Raylib
    return sound;
}

void BeepDependsOnExitDistance(Sound beep, int px, int py)
{
    if (gameState != STATE_PLAY)
        return;

    // 1. Trouver les coordonnées de la sortie (à faire une fois au GenerateMap)
    // Supposons exitX et exitY stockés globalement
    distExit = sqrtf(powf(exitX - px, 2) + powf(exitY - py, 2));

    // 2. Timer pour le bip
    static float timer = 0.0f;
    timer += GetFrameTime();

    // Plus on est proche, plus le délai est court (ex: entre 0.1s et 1.0s)
    float delay = 0.1f + (distExit * 0.05f); 

    if (timer >= delay) {
        // Ajuster le pitch (la hauteur) : plus proche = plus aigu
        float pitch = 2.0f - (distExit * 0.05f); 
        if (pitch < 0.5f) pitch = 0.5f;
        
        if(playBeepSound){
            SetSoundPitch(beep, pitch);
            PlaySound(beep);
        }
        timer = 0.0f;

        pulseRadius = 20.0f; 
    }
}
#pragma endregion


#pragma region KEY AND GAMEPAD
void KeysAndJoypadHandler(float* angle, float* px, float* py, float dt)
{
    if (gameState == STATE_WIN || gameState == STATE_LOST|| gameState == STATE_LOST_BY_CHASING
         || gameState == STATE_MAZE_NOT_READY) return;

    // ----- Déplacements -----
    float dx = cosf(*angle) * MOVE_SPEED * dt;
    float dy = sinf(*angle) * MOVE_SPEED * dt;

    if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
        // Avancer — vérifier collision avant de bouger
        if (MAP[(int)*py][(int)(*px + dx)] != 1) *px += dx;
        if (MAP[(int)(*py + dy)][(int)*px] != 1) *py += dy;
    }
    if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
        if (MAP[(int)*py][(int)(*px - dx)] != 1) *px -= dx;
        if (MAP[(int)(*py - dy)][(int)*px] != 1) *py -= dy;
    }

    // ----- Rotation -----
    if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A))
        *angle -= ROT_SPEED * dt;
    if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        *angle += ROT_SPEED * dt;
    
    // ----- Manette -----
    if (IsGamepadAvailable(0))
    {
        #define PITCH_SPEED  800.0f
        #define PITCH_MAX    300.0f

        // Joystick gauche — déplacement
        float axisX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X);
        float axisY = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y);

        // Joystick droit — rotation
        float axisRX = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_X);

        // Zone morte (évite les drifts)
        const float DEADZONE = 0.25f;

        // Avancer/reculer (axe Y du joystick gauche)
        if (fabsf(axisY) > DEADZONE) {
            float mdx = cosf(*angle) * (-axisY) * MOVE_SPEED * dt;
            float mdy = sinf(*angle) * (-axisY) * MOVE_SPEED * dt;
            if (MAP[(int)*py][(int)(*px + mdx)] != 1) *px += mdx;
            if (MAP[(int)(*py + mdy)][(int)*px] != 1) *py += mdy;
        }

        // Strafe (axe X du joystick gauche)
        if (fabsf(axisX) > DEADZONE) {
            float sdx = cosf(*angle + PI/2) * axisX * MOVE_SPEED * dt;
            float sdy = sinf(*angle + PI/2) * axisX * MOVE_SPEED * dt;
            if (MAP[(int)*py][(int)(*px + sdx)] != 1) *px += sdx;
            if (MAP[(int)(*py + sdy)][(int)*px] != 1) *py += sdy;
        }

        // Rotation (joystick droit)
        if (fabsf(axisRX) > DEADZONE)
            *angle += axisRX * ROT_SPEED * dt;

        // Joystick droit vertical
        float axisRY = GetGamepadAxisMovement(0, GAMEPAD_AXIS_RIGHT_Y);
        if (fabsf(axisRY) > DEADZONE)
            pitch += axisRY * PITCH_SPEED * dt;

        // Clamp
        if (pitch >  PITCH_MAX) pitch =  PITCH_MAX;
        if (pitch < -PITCH_MAX) pitch = -PITCH_MAX;
    }

    if (IsKeyPressed(KEY_R)) {
        //GenerateMapRandomWalk();
        //for (int y=0; y<MAP_W; y++)
        //    for (int x=0; x<MAP_H; x++)
        //        TRACES[x][y] = 0;
        // *px = 1.5f; *py = 1.5f;
        //initLight();
        ResetGame(px, py);
    }

    if (IsKeyPressed(KEY_R) && IsKeyDown(KEY_LEFT_SHIFT)) {
        GenerateMapWolfenstein();
        *px = 1.5f; *py = 1.5f;
    }

    if (IsKeyPressed(KEY_B))
        afficherMap = !afficherMap;

    if (IsKeyPressed(KEY_P))
        playBeepSound = !playBeepSound;

    // --- Contrôle de la Torche ---
    // On utilise le bouton Y (ou triangle) par exemple
    if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_UP)) 
    {
        torcheOn = !torcheOn; // Alterne entre true et false
    }

    if (IsKeyPressed(KEY_X) || IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_DOWN)) {
        if (solveMaze){
            memset(TRACES, 0, sizeof(TRACES));
            SolveMaze((int)*px, (int)*py);
        }
        else
            memset(TRACES, 0, sizeof(TRACES));

        solveMaze = !solveMaze;
    }

    if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_LEFT)) {
        //memset(TRACES, 0, sizeof(TRACES));

        if (deadendBranchOn)
        {
            for (int y=0; y<MAP_W; y++)
                for (int x=0; x<MAP_H; x++)
                    if (DEADEND_BRANCH[y][x]) TRACES[y][x] = 3;
        }
        else 
            for (int y=0; y<MAP_W; y++)
                for (int x=0; x<MAP_H; x++)
                    if (DEADEND_BRANCH[y][x]) TRACES[y][x] = 0;

        deadendBranchOn = !deadendBranchOn;
    }

    if (IsGamepadButtonPressed(0, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT)) {
        //memset(TRACES, 0, sizeof(TRACES));
        traceOn = !traceOn;
    }

    if (IsKeyDown(KEY_KP_ADD)) parallaxScale += 0.01f;
    if (IsKeyDown(KEY_KP_SUBTRACT)) parallaxScale -= 0.01f;

    if (parallaxScale < 0.0f) parallaxScale = 0.0f;
    if (parallaxScale > 2.0f) parallaxScale = 2.0f;
}
#pragma endregion


#pragma region RENDU

static inline float SampleHeight(Color* tex, int texW, int texH, float u, float v)
{
    u -= floorf(u);
    v -= floorf(v);

    int x = (int)(u * texW);
    int y = (int)(v * texH);

    if (x >= texW) x = texW - 1;
    if (y >= texH) y = texH - 1;

    return tex[y * texW + x].r / 255.0f;
}

// =============================================================
//  Fonction de rendu à utiliser dans RenderFrame
// =============================================================

void RenderWalls(Context* ctx, int startX, int endX, float px, float py, float angle, int horizon)
{
    for (int col = startX; col < endX; col += COL_STEP)
    {
        float ray_angle = angle + (col - renderW / 2.0f) * (FOV / renderW);

        #if RAYCASTER_TYPE
            RayHit hit = cast_ray_dda(px, py, ray_angle, angle);
        #else
            RayHit hit = cast_ray_raymarching(px, py, ray_angle, angle);
        #endif

        ctx->zBuffer[col] = hit.dist;
        for (int step = 1; step < COL_STEP && (col + step) < renderW; step++)
            ctx->zBuffer[col + step] = hit.dist;

        int wall_h_real = (int)(renderH / hit.dist);
        int wall_top    = horizon - wall_h_real / 2;
        int wall_bot    = horizon + wall_h_real / 2;
        int draw_top    = wall_top < 0       ? 0       : wall_top;
        int draw_bot    = wall_bot > renderH ? renderH : wall_bot;

        // ── Pré-calcul lumières ──────────────────────────────────────────
        typedef struct { float ldx, ldy, ldz, atten, r, g, b; int active; } LightContrib;
        LightContrib lc[8];
        for (int i = 0; i < numLights; i++) {
            lc[i].active = 0;
            float ldx   = lights[i].x - hit.wx;
            float ldy   = lights[i].y - hit.wy;
            float dist2 = ldx*ldx + ldy*ldy;
            if (dist2 > lights[i].radius * lights[i].radius) continue;

            float dist = dist2 * FastInvSqrt(dist2);
            float invLenL = FastInvSqrt(dist2 + 0.25f); // 0.5f * 0.5f = 0.25f

            lc[i].ldx = ldx * invLenL;
            lc[i].ldy = ldy * invLenL;
            lc[i].ldz = 0.5f * invLenL;
            lc[i].r   = lights[i].r;
            lc[i].g   = lights[i].g;
            lc[i].b   = lights[i].b;

            float Kc = 1.0f;
            float Kl = 2.0f / lights[i].radius;
            float Kq = 7.0f / (lights[i].radius * lights[i].radius);
            float atten = 1.0f / (Kc + Kl * dist + Kq * dist2);
            float fade  = 1.0f - (dist / lights[i].radius);
            lc[i].atten  = atten * (fade > 0.0f ? fade : 0.0f);
            lc[i].active = 1;
        }

        // ── [OPTIMISATION COLONNE] Pré-calculs POM Horizontaux ───────────
        // vue pixel -> espace caméra (composantes X et Z constantes sur la colonne)
        float viewX = LUTcos(ray_angle);
        float viewZ = LUTsin(ray_angle);

        // Tangente depuis la normale stockée dans hit
        float tangentX = -hit.ny;
        float tangentZ =  hit.nx;

        // Tangent space (X et Z constants sur la colonne)
        float vViewTS_x = viewX * tangentX + viewZ * tangentZ;
        float vViewTS_z = viewX * hit.nx   + viewZ * hit.ny;

        float faceSignX = 1.0f;
        if (hit.side == 0 && hit.nx < 0) faceSignX =  1.0f;
        if (hit.side == 0 && hit.nx > 0) faceSignX = -1.0f;
        if (hit.side == 1 && hit.ny < 0) faceSignX = -1.0f;  
        if (hit.side == 1 && hit.ny > 0) faceSignX =  1.0f;  

        float faceSignY = -1.0f;
        float safeZ = fmaxf(fabsf(vViewTS_z), 0.05f);

        typedef struct { float x; float y; } Vec2;

        // ── Boucle pixel ────────────────────────────────────────────────
        for (int y = draw_top; y < draw_bot; y++)
        {
            float texPos = fmodf((float)(y - wall_top) / (float)wall_h_real * TEX_TILE, 1.0f);

            // ── Steep Parallax Occlusion Mapping réorienté ────────────────
            float screenY = ((float)y - (wall_top + wall_h_real * 0.5f)) / (wall_h_real * 0.5f);
            float viewY = screenY;

            // Normalisation 3D rapide (viewX et viewZ sont réutilisés)
            float dot = viewX*viewX + viewY*viewY + viewZ*viewZ;
            float invLen = FastInvSqrt(dot);

            // Seul viewY (vViewTS_y) dépend réellement de 'y' après normalisation complète
            float normViewX = viewX * invLen;
            float normViewY = viewY * invLen;
            float normViewZ = viewZ * invLen;

            // Recalcul local à cause du changement de normalisation 3D par pixel
            float localTS_x = normViewX * tangentX + normViewZ * tangentZ;
            float localTS_z = normViewX * hit.nx   + normViewZ * hit.ny;
            float localSafeZ = fmaxf(fabsf(localTS_z), 0.05f);

            float vParallaxOffsetX = (localSafeZ > 0.01f) ? (localTS_x / localSafeZ) * parallaxScale : 0.0f;
            vParallaxOffsetX = fmaxf(-8.0f, fminf(8.0f, vParallaxOffsetX));

            int numLayers = PARALLAX_STEPS_MIN + (int)(fabsf(vParallaxOffsetX) * PARALLAX_STEPS_MAX);
            if (numLayers > 250) numLayers = 250;

            float layerDepth = 1.0f / numLayers;
            
            // DeltaTex final par pixel avec la normalisation corrigée
            Vec2 deltaTex = {
                faceSignX * localTS_x * parallaxScale / (localSafeZ * numLayers),
                faceSignY * normViewY * parallaxScale / (localSafeZ * numLayers)
            };

            float currentTexX     = hit.texX;
            float currentTexY     = texPos;
            float currentHeight   = 1.0f;
            float currentLayerDepth = 1.0f;
            float prevTexX        = currentTexX;
            float prevTexY        = currentTexY;
            float prevHeight      = 1.0f;
            float prevLayerDepth  = 1.0f;

            // Marche de rayons (Raymarching de la Heightmap)
            for (int i = 0; i < numLayers; i++)
            {
                prevTexX        = currentTexX;
                prevTexY        = currentTexY;
                prevHeight      = currentHeight;
                prevLayerDepth  = currentLayerDepth;

                currentTexX -= deltaTex.x;
                currentTexY -= deltaTex.y;

                float wrappedX = currentTexX - floorf(currentTexX);
                float wrappedY = currentTexY - floorf(currentTexY);

                int sampleX = (int)(wrappedX * ctx->texW);
                int sampleY = (int)(wrappedY * ctx->texH);

                sampleX = fmaxf(0, fminf(ctx->texW - 1, sampleX));
                sampleY = fmaxf(0, fminf(ctx->texH - 1, sampleY));

                currentHeight      = ctx->wallHeight[sampleY * ctx->texW + sampleX].r / 255.0f;
                currentLayerDepth -= layerDepth;

                if (currentHeight >= currentLayerDepth) break;
            }

            // ── Interpolation POM ────────────────────────────────────────
            float afterDepth  = currentHeight - currentLayerDepth;
            float beforeDepth = prevHeight    - prevLayerDepth;
            float denom  = afterDepth - beforeDepth;
            float weight = (fabsf(denom) > 0.0001f) ? afterDepth / denom : 0.0f;
            weight = fmaxf(0.0f, fminf(1.0f, weight));

            float finalTexXf = prevTexX * weight + currentTexX * (1.0f - weight);
            finalTexXf -= floorf(finalTexXf);

            float finalTexYf = prevTexY * weight + currentTexY * (1.0f - weight);
            finalTexYf -= floorf(finalTexYf);

            // ── Échantillonnage bilinéaire ───────────────────────────────
            Color s = SampleBilinear(ctx->wallPixels, ctx->texW,  ctx->texH,  finalTexXf, finalTexYf);
            Color n = SampleBilinear(ctx->wallNormal, ctx->normW, ctx->normH, finalTexXf, finalTexYf);

            // ── Normal mapping ───────────────────────────────────────────
            float nnx = (n.r / 255.0f) * 2.0f - 1.0f;
            float nny = (n.g / 255.0f) * 2.0f - 1.0f;
            float nnz = (n.b / 255.0f) * 2.0f - 1.0f;

            float outR = AMBIENT_LIGHT, outG = AMBIENT_LIGHT, outB = AMBIENT_LIGHT;

            // Boucle d'éclairage et self-shadowing 
            for (int i = 0; i < numLights; i++) {
                if (!lc[i].active) continue;

                float finalHeight = SampleHeight(ctx->wallHeight, ctx->heightW, ctx->heightH, finalTexXf, finalTexYf);
                
                float ldx = lc[i].ldx;
                float ldy = lc[i].ldy;
                float ldz = lc[i].ldz; 

                float lightTexX = ldx * tangentX + ldy * tangentZ;
                float lightTexY = -ldz; 
                float lightWallDepth = ldx * hit.nx + ldy * hit.ny;
                float safeDepth = fmaxf(fabsf(lightWallDepth), 0.05f);

                Vec2 shadowDelta = {
                    (lightTexX / safeDepth) * parallaxScale,
                    (lightTexY / safeDepth) * parallaxScale
                };

                shadowDelta.x = fmaxf(-0.5f, fminf(0.5f, shadowDelta.x));
                shadowDelta.y = fmaxf(-0.5f, fminf(0.5f, shadowDelta.y));

                float heightStep = 1.0f / (float)SHADOW_STEPS;
                float shadowBias = 0.02f; 

                float shadowTexX   = finalTexXf + shadowDelta.x * shadowBias;
                float shadowTexY   = finalTexYf + shadowDelta.y * shadowBias;
                float shadowHeight = finalHeight + shadowBias;
                float shadow = 1.0f;

                for(int s_step = 0; s_step < SHADOW_STEPS; s_step++)
                {
                    shadowTexX += shadowDelta.x * heightStep;
                    shadowTexY += shadowDelta.y * heightStep;
                    shadowHeight += heightStep;

                    if(shadowHeight >= 1.0f) break; 

                    float u = shadowTexX - floorf(shadowTexX);
                    float v = shadowTexY - floorf(shadowTexY);
                    float sampleH = SampleHeight(ctx->wallHeight, ctx->heightW, ctx->heightH, u, v);

                    if(sampleH > shadowHeight)
                    {
                        float distanceIntoWall = sampleH - shadowHeight;
                        float distanceFactor = 1.0f - (float)s_step / SHADOW_STEPS;
                        //float distanceFactor = 1.0f;
                        shadow = fmaxf(0.35f, 1.0f - (distanceIntoWall * 5.0f * distanceFactor)); 
                        break;
                    }
                }

                float diff = fmaxf(0.0f, nnx*ldx + nny*ldy + nnz*ldz);
                outR += diff * shadow * lc[i].atten * lc[i].r;
                outG += diff * shadow * lc[i].atten * lc[i].g;
                outB += diff * shadow * lc[i].atten * lc[i].b;
            }

            // ── Couleur finale ───────────────────────────────────────────
            Color finalColor = s;
            if (hit.wallType == 3) finalColor = MixColor(s, WHITE, 0.5f);
            if (hit.wallType == 4) finalColor = MixColor(s, RED,   0.5f);
            if (hit.wallType == 5) finalColor = MixColor(s, GREEN, 0.5f);
            if (hit.wallType == 2) finalColor = MixColor(s, GOLD,  0.7f);

            for (int step = 0; step < COL_STEP && (col + step) < renderW; step++) {
                ctx->framebuffer[y * renderW + (col + step)] = (Color){
                    (unsigned char)fminf(255.0f, finalColor.r * outR),
                    (unsigned char)fminf(255.0f, finalColor.g * outG),
                    (unsigned char)fminf(255.0f, finalColor.b * outB),
                    255
                };
            }
        }
    }
}


// =============================================================
//  Fonction de rendu à utiliser dans RenderFrame
// =============================================================

void RenderFloorCeil(Context* ctx, float px, float py, float angle, int horizon){
    memset(ctx->framebuffer, 0, renderW * renderH * sizeof(Color));
    // Sol/plafond par ligne
    for (int y = 0; y < renderH; y++)
    {
        bool is_floor = (y > horizon);
        float row_dist = (float)halfRenderH / fabsf(y - horizon + 0.0001f);
        float fog = 1.0f - fminf(row_dist / 12.0f, 1.0f);

        // Position centrale de la ligne (pour calculer l'éclairage de la ligne)
        float floor_cx = px + LUTcos(angle) * row_dist;
        float floor_cy = py + LUTsin(angle) * row_dist;

        float fR = AMBIENT_LIGHT, fG = AMBIENT_LIGHT, fB = AMBIENT_LIGHT;

        for (int i = 0; i < numLights; i++) {
            float ldx   = lights[i].x - floor_cx;
            float ldy   = lights[i].y - floor_cy;
            float dist2 = ldx*ldx + ldy*ldy;

            if (dist2 > lights[i].radius * lights[i].radius) continue;

            float dist = sqrtf(dist2);

            // --- MÊME FORMULE QUE POUR LES MURS ---
            float Kc = 1.0f;
            float Kl = 2.0f / lights[i].radius;
            float Kq = 7.0f / (lights[i].radius * lights[i].radius);

            float atten = 1.0f / (Kc + Kl * dist + Kq * dist2);
            
            // Soft fade à la limite du rayon pour éviter les bords nets
            float fade = 1.0f - (dist / lights[i].radius);
            float finalAtten = atten * (fade > 0 ? fade : 0);

            fR += lights[i].r * finalAtten;
            fG += lights[i].g * finalAtten;
            fB += lights[i].b * finalAtten;
        }
        
        // On sature à 1.0 après avoir accumulé toutes les lumières
        fR = fminf(1.0f, fR); fG = fminf(1.0f, fG); fB = fminf(1.0f, fB);

        float rayL = angle - FOV / 2.0f;
        float rayR = angle + FOV / 2.0f;
        float floorStepX = (LUTcos(rayR) - LUTcos(rayL)) * row_dist / renderW;
        float floorStepY = (LUTsin(rayR) - LUTsin(rayL)) * row_dist / renderW;
        float floorX = px + LUTcos(rayL) * row_dist;
        float floorY = py + LUTsin(rayL) * row_dist;

        for (int x = 0; x < renderW; x++, floorX += floorStepX, floorY += floorStepY)
        {
            int tx, ty;
            Color base;
            if (is_floor) {
                tx = (int)(fabsf(fmodf(floorX, 1.0f)) * ctx->texFloorW) % ctx->texFloorW;
                ty = (int)(fabsf(fmodf(floorY, 1.0f)) * ctx->texFloorH) % ctx->texFloorH;
                base = ctx->floorPixels[ty * ctx->texFloorW + tx];
            } else {
                tx = (int)(fabsf(fmodf(floorX, 1.0f)) * ctx->texCeilW) % ctx->texCeilW;
                ty = (int)(fabsf(fmodf(floorY, 1.0f)) * ctx->texCeilH) % ctx->texCeilH;
                base = ctx->ceilPixels[ty * ctx->texCeilW + tx];
            }
            ctx->framebuffer[y * renderW + x] = (Color){
                (unsigned char)(base.r * fR * fog),
                (unsigned char)(base.g * fG * fog),
                (unsigned char)(base.b * fB * fog),
                255
            };
        }
    }
}


// =============================================================
//  Fonction de rendu de sprites
// =============================================================

void RenderSprites(Context* ctx, float px, float py, float angle)
{
    memset(ctx->framebuffersprites, 0, renderW * renderH * sizeof(Color));
    // Mettre à jour positions EN CONSERVANT l'index
    for (int i = 0; i < PATROL_LIGHTS; i++) {
        sprites[i].x        = patrolLights[i].l->x;  // ← toujours patrolLights[i]
        sprites[i].y        = patrolLights[i].l->y;
        sprites[i].lightIdx = i;                       // ← index fixe
        float dx = sprites[i].x - px;
        float dy = sprites[i].y - py;
        sprites[i].dist = dx*dx + dy*dy;
    }

    // Tri à bulle par distance décroissante pour N sprites
    int order[PATROL_LIGHTS];
    for (int i = 0; i < PATROL_LIGHTS; i++) order[i] = i;

    for (int i = 0; i < PATROL_LIGHTS - 1; i++)
        for (int j = 0; j < PATROL_LIGHTS - 1 - i; j++)
            if (sprites[order[j]].dist < sprites[order[j+1]].dist) {
                int tmp = order[j];
                order[j] = order[j+1];
                order[j+1] = tmp;
            }

    for (int si = 0; si < PATROL_LIGHTS; si++)
    {
        int s = order[si];

        float sx = sprites[s].x - px;
        float sy = sprites[s].y - py;

        // Angle du sprite par rapport au joueur
        float spriteAngle = atan2f(sy, sx);

        // Différence d'angle entre la direction du joueur et le sprite
        float angleDiff = spriteAngle - angle;

        // Normaliser entre -PI et PI
        while (angleDiff >  PI) angleDiff -= 2.0f * PI;
        while (angleDiff < -PI) angleDiff += 2.0f * PI;

        // Hors du champ de vision → skip
        if (fabsf(angleDiff) > FOV / 2.0f + 0.1f) continue;

        float spriteDist = sqrtf(sprites[s].dist);
        if (spriteDist < 0.1f) continue;

        // Position X écran basée sur l'angle
        int spriteScreenX = (int)((renderW / 2) + (angleDiff / (FOV / 2.0f)) * (renderW / 2));

        // Taille à l'écran basée sur la distance perpendiculaire
        int spriteSize = abs((int)(renderH / spriteDist));

        int drawStartY = halfRenderH - spriteSize / 2 + (int)pitch;
        int drawEndY   = halfRenderH + spriteSize / 2 + (int)pitch;
        if (drawStartY < 0)        drawStartY = 0;
        if (drawEndY >= renderH)   drawEndY   = renderH - 1;

        int drawStartX = spriteScreenX - spriteSize / 2;
        int drawEndX   = spriteScreenX + spriteSize / 2;
        if (drawStartX < 0)        drawStartX = 0;
        if (drawEndX >= renderW)   drawEndX   = renderW - 1;

        Color tint;
        if (patrolLights[s].chasing)
            tint = (Color){255, 0, 0, 255};
        else
        /*
            tint = (sprites[s].lightIdx == 0)
                ? (Color){255, 200,  50, 255}
                : (Color){ 50, 150, 255, 255};
        */
            tint = (Color){(int)(fminf(patrolLights[s].l->r, 1.0f) * 255.0), 
                           (int)(fminf(patrolLights[s].l->g, 1.0f) * 255.0), 
                           (int)(fminf(patrolLights[s].l->b, 1.0f) * 255.0), 255};

        for (int pass = 0; pass <= BLOOM_PASSES; pass++)
        {
            // Pass 0 = sprite normal, passes suivantes = halo décalé
            int offsets[8][2] = {
                {0,0},
                {-BLOOM_SPREAD, 0}, {BLOOM_SPREAD, 0},
                {0, -BLOOM_SPREAD}, {0,  BLOOM_SPREAD},
                {-BLOOM_SPREAD,-BLOOM_SPREAD}, {BLOOM_SPREAD,-BLOOM_SPREAD},
                {-BLOOM_SPREAD, BLOOM_SPREAD}
            };
            int ox = (pass == 0) ? 0 : offsets[pass][0];
            int oy = (pass == 0) ? 0 : offsets[pass][1];
            float passAlpha = (pass == 0) ? 1.0f : BLOOM_ALPHA;

            for (int col = drawStartX; col < drawEndX; col++)
            {
                if (col < 0 || col >= renderW) continue;

                int texX = (int)((float)(col - drawStartX) / (float)(drawEndX - drawStartX) * ctx->spriteW);
                if (texX < 0) texX = 0;
                if (texX >= ctx->spriteW) texX = ctx->spriteW - 1;

                if (spriteDist >= ctx->zBuffer[col]) continue;

                for (int row = drawStartY; row < drawEndY; row++)
                {
                    int texY = (int)((float)(row - drawStartY) / (float)(drawEndY - drawStartY) * ctx->spriteH);
                    if (texY < 0) texY = 0;
                    if (texY >= ctx->spriteH) texY = ctx->spriteH - 1;

                    Color c = ctx->spritePixels[texY * ctx->spriteW + texX];
                    if (c.a < 128) continue;

                    float fog = 1.0f - fminf(spriteDist / 12.0f, 1.0f);

                    int destCol = col + ox;
                    int destRow = row + oy;
                    if (destCol < 0 || destCol >= renderW) continue;
                    if (destRow < 0 || destRow >= renderH) continue;

                    // Pour le halo : additionner sur le framebuffer existant
                    if (pass == 0) {
                        ctx->framebuffersprites[destRow * renderW + destCol] = (Color){
                            (unsigned char)(c.r * tint.r / 255.0f * fog),
                            (unsigned char)(c.g * tint.g / 255.0f * fog),
                            (unsigned char)(c.b * tint.b / 255.0f * fog),
                            255
                        };
                    } else {
                        // Additive blending sur le framebuffer
                        Color dst = ctx->framebuffersprites[destRow * renderW + destCol];
                        ctx->framebuffersprites[destRow * renderW + destCol] = (Color){
                            (unsigned char)fminf(255, dst.r + c.r * tint.r / 255.0f * fog * passAlpha),
                            (unsigned char)fminf(255, dst.g + c.g * tint.g / 255.0f * fog * passAlpha),
                            (unsigned char)fminf(255, dst.b + c.b * tint.b / 255.0f * fog * passAlpha),
                            255
                        };
                    }
                }
            }
        }
    }
}


// =============================================================
//  Fonction de rendu de raycaster to pixel shader
// =============================================================
float* RenderWallPixelShader(Context* ctx, float px, float py, float angle, float* rayOffset, int horizon)
{
    int stride = renderW * 4;  // 4 floats par pixel (RGBA)

    for(int col = 0; col < renderW; col++)
    {
        float ray_angle = angle + rayOffset[col];
        RayHit hit = cast_ray_dda(px, py, ray_angle, angle);

        if(hit.dist<0.01f)
            hit.dist=0.01f;

        ctx->zBuffer[col]=hit.dist;

        // --- SÉCURITÉ DISTANCE (Anti Division par Zéro) ---
        if (hit.dist < 0.01f) hit.dist = 0.01f; 

        float *p0 = &ctx->wallDataBuf[col * 4];              // ligne 0
        float *p1 = &ctx->wallDataBuf[stride + col * 4];     // ligne 1
        float *p2 = &ctx->wallDataBuf[stride*2 + col * 4];   // ligne 2

        // Ligne 0
        p0[0] = hit.texX;
        p0[1] = hit.nx;
        p0[2] = hit.ny;
        p0[3] = hit.dist;

        // Ligne 1
        p1[0] = hit.wx;
        p1[1] = hit.wy;
        p1[2] = (float)hit.wallType;
        p1[3] = (float)hit.side;

        // Tangent space
        float vViewWS_x = hit.wx - px;
        float vViewWS_z = hit.wy - py;

        float tangentX = -hit.ny;
        float tangentZ =  hit.nx;

        float vViewTS_x =
            vViewWS_x*tangentX +
            vViewWS_z*tangentZ;

        float vViewTS_z =
            vViewWS_x*hit.nx +
            vViewWS_z*hit.ny;

        // offset brut
        float vParallaxOffsetTS = (fabsf(vViewTS_z)>0.01f)
        ? (vViewTS_x/vViewTS_z)*parallaxScale
        :0.0f;


        // évite les valeurs folles
        //vParallaxOffsetTS = fmaxf(-0.08f, fminf(0.08f,vParallaxOffsetTS));

        // correction orientation mur
        float faceSign=1.0f;

        if(hit.side==0 && hit.nx<0) faceSign= 1.0f;
        if(hit.side==0 && hit.nx>0) faceSign=-1.0f;
        if(hit.side==1 && hit.ny<0) faceSign=-1.0f;
        if(hit.side==1 && hit.ny>0) faceSign= 1.0f;

        float signedOffset= faceSign * (vViewTS_x>0?1.0f:-1.0f) * fabsf(vParallaxOffsetTS);

        // hauteur mur
        float wall_h=(float)renderH/hit.dist;
        float top = horizon - wall_h * 0.5f;
        float bot = horizon + wall_h * 0.5f;

        // ligne 2
        p2[0]=vViewTS_x;
        p2[1]=signedOffset;
        p2[2]=top;
        p2[3]=bot;
    }

    return ctx->wallDataBuf;
}
#pragma endregion


#pragma region ANIMATION LUMIERES
// =============================================================
//  Gestion de l'animation des lumières
// =============================================================

void AnimLights(float px, float py, float angle, float dt) {
    for (int i = 0; i < PATROL_LIGHTS; i++) {

        float dxp = patrolLights[i].l->x - px;
        float dyp = patrolLights[i].l->y - py;
        float distPlayer = sqrtf(dxp*dxp + dyp*dyp);

        // Bascule entre poursuite et patrouille
        if (distPlayer < CHASE_RADIUS && torcheOn)
            patrolLights[i].chasing = true;
        else if (distPlayer > CHASE_RADIUS + 2.0f || !torcheOn)  // hysteresis pour éviter le flicker
            patrolLights[i].chasing = false;

        float speed = patrolLights[i].chasing ? PATROL_LIGHT_SPEED * 2.0f : PATROL_LIGHT_SPEED;

        if (patrolLights[i].chasing){
            patrolLights[i].l->r = 1.0f; patrolLights[i].l->g = 0.1f; patrolLights[i].l->b = 0.1f;  // rouge
        }
        else {
            patrolLights[i].l->r = tmpLightColor[i].r; 
            patrolLights[i].l->g = tmpLightColor[i].g; 
            patrolLights[i].l->b = tmpLightColor[i].b;
        }

        // Cible BFS selon le mode
        int targetX = patrolLights[i].chasing ? (int)px : exitX;
        int targetY = patrolLights[i].chasing ? (int)py : exitY;

        float nextX = patrolLights[i].l->x + patrolLights[i].dx * dt * speed;
        float nextY = patrolLights[i].l->y + patrolLights[i].dy * dt * speed;

        int gx = (int)nextX;
        int gy = (int)nextY;

        bool collisionX = (gx < 0 || gx >= MAP_W || MAP[(int)patrolLights[i].l->y][gx] == 1);
        bool collisionY = (gy < 0 || gy >= MAP_H || MAP[gy][(int)patrolLights[i].l->x] == 1);

        if (collisionX || collisionY) {
            int cx = (int)patrolLights[i].l->x;
            int cy = (int)patrolLights[i].l->y;

            // Snap complet au centre pour repartir proprement
            patrolLights[i].l->x = cx + 0.5f;
            patrolLights[i].l->y = cy + 0.5f;

            Vec2 dir = GetNextDirToward(cx, cy, targetX, targetY);
            if (dir.x != 0 || dir.y != 0) {
                patrolLights[i].dx = dir.x;
                patrolLights[i].dy = dir.y;
            } else {
                patrolLights[i].dx *= -1;
                patrolLights[i].dy *= -1;
            }

            // Recalcul depuis la position snappée
            nextX = patrolLights[i].l->x + patrolLights[i].dx * dt * speed;
            nextY = patrolLights[i].l->y + patrolLights[i].dy * dt * speed;
            gx = (int)nextX;
            gy = (int)nextY;

            if (gx < 0 || gx >= MAP_W || MAP[(int)patrolLights[i].l->y][gx] == 1)
                nextX = patrolLights[i].l->x;
            if (gy < 0 || gy >= MAP_H || MAP[gy][(int)patrolLights[i].l->x] == 1)
                nextY = patrolLights[i].l->y;

            gx = (int)nextX;
            gy = (int)nextY;
        }

        // Changement de case
        if (gx != patrolLights[i].lastGridX || gy != patrolLights[i].lastGridY) {

            if (gx == exitX && gy == exitY)
                gameState = STATE_LOST;

            Vec2 dir = GetNextDirToward(gx, gy, targetX, targetY);
            if (dir.x != 0 || dir.y != 0) {
                patrolLights[i].dx = dir.x;
                patrolLights[i].dy = dir.y;
            }

            // Snap perpendiculaire DANS LA NOUVELLE CASE
            // On corrige l'axe qui ne bouge pas pour rester centré
            if (patrolLights[i].dx != 0)  // mouvement horizontal → centrer Y dans nouvelle case
                patrolLights[i].l->y = gy + 0.5f;
            if (patrolLights[i].dy != 0)  // mouvement vertical → centrer X dans nouvelle case
                patrolLights[i].l->x = gx + 0.5f;

            nextX = patrolLights[i].l->x + patrolLights[i].dx * dt * PATROL_LIGHT_SPEED;
            nextY = patrolLights[i].l->y + patrolLights[i].dy * dt * PATROL_LIGHT_SPEED;

            patrolLights[i].lastGridX = gx;
            patrolLights[i].lastGridY = gy;
        }

        patrolLights[i].l->x = nextX;
        patrolLights[i].l->y = nextY;

        if (patrolLights[i].chasing){
            // Détection collision avec le joueur par distance
            float cdx = patrolLights[i].l->x - px;
            float cdy = patrolLights[i].l->y - py;
            if (cdx*cdx + cdy*cdy < 0.5f * 0.5f)
                gameState = STATE_LOST_BY_CHASING;
        }
    }

    float offset = TORCHE_DISTANCE;
    if (torcheOn){
        lights[2].x = px + cosf(angle) * offset;
        lights[2].y = py + sinf(angle) * offset;
        lights[2].r = TORCHE_PUISSANCE; lights[2].g = TORCHE_PUISSANCE; lights[2].b = TORCHE_PUISSANCE;
        lights[2].radius = TORCHE_RADIUS;
    }
    else lights[2] = (Light){0};
}
#pragma endregion


#pragma region GAME STATE
// =============================================================
//  Machine à état pour le jeu
// =============================================================

void GState(GameState gameState, float* px, float* py, Sound tada, Sound siren){
    switch (gameState){
        case STATE_WIN:

            if (!IsSoundPlaying(tada) && playSoundOneTime) { PlaySound(tada); playSoundOneTime = !playSoundOneTime; };

            int seconds = (int)gameTimer;
            int minutes  = seconds / 60;
            seconds     %= 60;

            // Meilleur temps
            if (bestTime == 0 || (int)gameTimer < bestTime)
                bestTime = (int)gameTimer;

            int bMin = bestTime / 60;
            int bSec = bestTime % 60;

            char buf[64];
            sprintf(buf, "Temps : %02d:%02d", minutes, seconds);
            DrawText(buf, SCREEN_W/2 - 100, SCREEN_H/2,      30, YELLOW);

            sprintf(buf, "Meilleur : %02d:%02d", bMin, bSec);
            DrawText(buf, SCREEN_W/2 - 100, SCREEN_H/2 + 40, 20, WHITE);

            DrawText("VICTOIRE ! SORTIE TROUVÉE !",  SCREEN_W/2 - 200, SCREEN_H/2 - 50, 30, RED);
            DrawText("Appuyez sur R pour rejouer",   SCREEN_W/2 - 150, SCREEN_H/2 + 70, 20, WHITE);

            if (IsKeyPressed(KEY_R) || IsGamepadButtonPressed(0, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            {
                ResetGame(px, py);
            }
            break;

        case STATE_LOST:

            if (!IsSoundPlaying(siren) && playSoundOneTime) { PlaySound(siren); playSoundOneTime= !playSoundOneTime; };

            DrawText("UNE LUMIÈRE A TROUVÉ LA SORTIE !",  SCREEN_W/2 - 220, SCREEN_H/2 - 50, 30, RED);
            DrawText("Appuyez sur R pour rejouer", SCREEN_W/2 - 150, SCREEN_H/2 + 10, 20, WHITE);

            if (IsKeyPressed(KEY_R) || IsGamepadButtonPressed(0, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            {
                ResetGame(px, py);
            }
            break;

            case STATE_LOST_BY_CHASING:

            if (!IsSoundPlaying(siren) && playSoundOneTime) { PlaySound(siren); playSoundOneTime= !playSoundOneTime; };

            DrawText("UNE LUMIÈRE VOUS A MANGE !",  SCREEN_W/2 - 220, SCREEN_H/2 - 50, 30, RED);
            DrawText("Appuyez sur R pour rejouer", SCREEN_W/2 - 150, SCREEN_H/2 + 10, 20, WHITE);

            for (int i=0; i<PATROL_LIGHTS; i++){
                patrolLights[i].chasing = false;
            
                patrolLights[i].l->r = tmpLightColor[i].r;
                patrolLights[i].l->g = tmpLightColor[i].g;
                patrolLights[i].l->b = tmpLightColor[i].b;
            }

            if (IsKeyPressed(KEY_R) || IsGamepadButtonPressed(0, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            {
                ResetGame(px, py);
            }
            break;

        case STATE_MAZE_NOT_READY:
            DrawText("PAS DE SORTIE TROUVEE !",  SCREEN_W/2 - 220, SCREEN_H/2 - 50, 30, RED);
            DrawText("Appuyez sur R pour régénérer", SCREEN_W/2 - 150, SCREEN_H/2 + 10, 20, WHITE);

            if (IsKeyPressed(KEY_R) || IsGamepadButtonPressed(0, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            {
                ResetGame(px, py);
            }        
            break;
        
        default:
            break;
    }
}
#pragma endregion


#pragma region MAIN
// =============================================================
//  Programme principal
// =============================================================

int main(void)
{
    InitLUT();

    SCREEN_W = 1920;
    SCREEN_H = 1080;
    HALF_H   = SCREEN_H / 2;

    SetConfigFlags(FLAG_FULLSCREEN_MODE);
    InitWindow(SCREEN_W, SCREEN_H, "Raycaster - Raylib 6.0");

    HideCursor();

    // Calcul de la résolution de rendu
    renderW = SCREEN_W / RENDER_SCALE;
    renderH = SCREEN_H / RENDER_SCALE;
    //renderW = 1280;
    //renderH = 720;
    halfRenderH = renderH / 2;

    Image imgWall = LoadImage("Wall2.png");
    ImageFormat(&imgWall, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D texDiffuse = LoadTextureFromImage(imgWall);
    int texW = imgWall.width;
    int texH = imgWall.height;
    Color *wallPixels = LoadImageColors(imgWall);
    UnloadImage(imgWall);

    Image imgNorm = LoadImage("Wall2_norm.png");
    ImageFormat(&imgNorm, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D texNormal = LoadTextureFromImage(imgNorm);
    int normW = imgNorm.width;
    int normH = imgNorm.height;
    Color *wallNormal = LoadImageColors(imgNorm);
    UnloadImage(imgNorm);

    Image imgHeight = LoadImage("Wall2_disp.png");
    ImageFormat(&imgHeight, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D texHeight = LoadTextureFromImage(imgHeight);
    int heightW = imgHeight.width;
    int heightH = imgHeight.height;
    //ImageColorInvert(&imgHeight);
    Color *wallHeight = LoadImageColors(imgHeight);
    UnloadImage(imgHeight);

    Image imgFloor = LoadImage("Floor.png");
    ImageFormat(&imgFloor, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    int texFloorW = imgFloor.width;
    int texFloorH = imgFloor.height;
    Color *floorPixels = LoadImageColors(imgFloor);
    UnloadImage(imgFloor);

    Image imgCeil = LoadImage("Ceil.png");
    ImageFormat(&imgCeil, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    int texCeilW = imgCeil.width;
    int texCeilH = imgCeil.height;
    Color *ceilPixels = LoadImageColors(imgCeil);
    UnloadImage(imgCeil);

    Image imgSprite = LoadImage("Ghost.png");  // ton sprite
    //Image imgSprite = LoadImage("Flame.png");  // ton sprite
    ImageFormat(&imgSprite, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    int spriteW = imgSprite.width;
    int spriteH = imgSprite.height;
    Color *spritePixels = LoadImageColors(imgSprite);
    UnloadImage(imgSprite);

    Color *framebuffer = malloc(renderW * renderH * sizeof(Color));
    Color *framebuffersprites = malloc(renderW * renderH * sizeof(Color));
    float *zBuffer = malloc(renderW * sizeof(float));
    Color *tmpFxaa = malloc(renderW * renderH * sizeof(Color));

    Image img = GenImageColor(renderW, renderH, BLACK);
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D screenTex = LoadTextureFromImage(img);
    UnloadImage(img);

    // Textures à créer pour pixel shader
    // 1. On crée un tableau de float vide (3 lignes de pixels, chaque pixel a 4 canaux R, G, B, A)
    float *emptyData = calloc(renderW * 3 * 4, sizeof(float));

    Image wallDataImg = {
        .data = emptyData,
        .width = renderW,   // Largeur = nombre de colonnes de votre raycaster
        .height = 3,        // Hauteur = 3 lignes (a, b, c pour texelFetch)
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R32G32B32A32 // <- Crucial pour accepter les float CPU !
    };

    Texture2D wallDataTex = LoadTextureFromImage(wallDataImg);

    // Forcer le mode d'échantillonnage brut sans lissage (indispensable pour texelFetch)
    SetTextureFilter(wallDataTex, TEXTURE_FILTER_POINT);
    SetTextureWrap(wallDataTex, TEXTURE_WRAP_CLAMP);

    free(emptyData); // On libère la mémoire CPU temporaire

    // Buffer CPU — 1 pixel par colonne
    float *wallDataBuf = malloc(renderW * 12 * sizeof(float));


    // Initialisation du context
    Context ctx;
    ctx.texDiffuse = texDiffuse;
    ctx.texNormal = texNormal;
    ctx.texHeight = texHeight;
    ctx.wallDataTex = wallDataTex;
    ctx.wallDataBuf = wallDataBuf;
    ctx.framebuffer = framebuffer, ctx.framebuffersprites = framebuffersprites, ctx.tmpFxaa = tmpFxaa;
    ctx.screenTex = screenTex;
    ctx.wallPixels = wallPixels;
    ctx.texW = texW, ctx.texH = texH;
    ctx.wallNormal = wallNormal;
    ctx.normW = normW;
    ctx.normH = normH;
    ctx.wallHeight = wallHeight;
    ctx.heightW = heightW, ctx.heightH = heightH;
    ctx.floorPixels = floorPixels;
    ctx.texFloorW = texFloorW, ctx.texFloorH = texFloorH;
    ctx.ceilPixels = ceilPixels;
    ctx.texCeilW = texCeilW, ctx.texCeilH = texCeilH;
    ctx.spritePixels = spritePixels;
    ctx.spriteW = spriteW;
    ctx.spriteH = spriteH;
    ctx.zBuffer = zBuffer;

    InitAudioDevice(); // <--- Très important !

    // On crée les sons pour le jeu
    Sound beep  = CreateBeep(); 
    Sound tada  = CreateTada();
    Sound siren = CreateSiren();  

    SetTargetFPS(60);

    // Position et angle de départ du joueur
    float px    = 1.5f;
    float py    = 1.5f;
    float angle = 0.0f;
    
    ResetGame(&px, &py);

    // Shader
    Shader shader = LoadShader(NULL, "shaders/raycaster.fs");

    shader.locs[SHADER_LOC_MAP_DIFFUSE]   = GetShaderLocation(shader, "u_diffuseTexture");
    shader.locs[SHADER_LOC_MAP_NORMAL]    = GetShaderLocation(shader, "u_normalTexture");
    shader.locs[SHADER_LOC_MAP_ROUGHNESS] = GetShaderLocation(shader, "u_heightTexture");

    SetTextureFilter(texDiffuse, TEXTURE_FILTER_POINT);
    SetTextureFilter(texNormal,  TEXTURE_FILTER_POINT);
    SetTextureFilter(texHeight,  TEXTURE_FILTER_POINT);
    
    SetTextureWrap(texDiffuse, TEXTURE_WRAP_REPEAT);
    SetTextureWrap(texNormal, TEXTURE_WRAP_REPEAT);
    SetTextureWrap(texHeight, TEXTURE_WRAP_REPEAT);

    int locWallData        = GetShaderLocation(shader, "wallData");
    int locDiffuse         = GetShaderLocation(shader, "u_diffuseTexture");
    int locNormal          = GetShaderLocation(shader, "u_normalTexture");
    int locHeight          = GetShaderLocation(shader, "u_heightTexture");
    int locLightPos        = GetShaderLocation(shader, "lightPos");
    int locLightColor      = GetShaderLocation(shader, "lightColor");
    int locLightRadius     = GetShaderLocation(shader, "lightRadius");
    int locNumLights       = GetShaderLocation(shader, "numLights");
    int locResolution      = GetShaderLocation(shader, "resolution");
    int locHorizon         = GetShaderLocation(shader, "horizon");
    int locParallaxScale   = GetShaderLocation(shader, "parallaxScale");
    int locMinSamples      = GetShaderLocation(shader, "g_nMinSamples");
    int locMaxSamples      = GetShaderLocation(shader, "g_nMaxSamples");
    int locTiling          = GetShaderLocation(shader, "tiling");
    int locNormalStrength  = GetShaderLocation(shader, "normalStrength");
    int locPlayer          = GetShaderLocation(shader, "playerPos");

    SetShaderValue(shader, locDiffuse,          (int[1]){0}, SHADER_UNIFORM_INT);
    SetShaderValue(shader, locNormal,           (int[1]){1}, SHADER_UNIFORM_INT);
    SetShaderValue(shader, locHeight,           (int[1]){2}, SHADER_UNIFORM_INT);
    float tiling[2] = {TEX_TILE,TEX_TILE};
    SetShaderValue(shader, locTiling, tiling,                SHADER_UNIFORM_VEC2);

    float *rayOffset=malloc(renderW*sizeof(float));

    for(int x=0;x<renderW;x++)
    {
        rayOffset[x]= (x-renderW*0.5f)*(FOV/renderW);
    }

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime();

        if (gameState == STATE_PLAY)
            gameTimer += dt;

        // ----- Animation des lumières -----
        AnimLights(px, py, angle, dt);
 
        float lightPositions[NUM_LIGHTS * 3];
        float lightColors[NUM_LIGHTS * 3];
        float lightRadii[NUM_LIGHTS];

        for (int i = 0; i < NUM_LIGHTS; i++)
        {
            lightPositions[i*3+0] = lights[i].x;
            lightPositions[i*3+1] = lights[i].y;
            lightPositions[i*3+2] = 0.5f;  // hauteur fixe

            lightColors[i*3+0] = lights[i].r;
            lightColors[i*3+1] = lights[i].g;
            lightColors[i*3+2] = lights[i].b;

            lightRadii[i] = lights[i].radius;
        }
        // ------------------------------------- 

        BeepDependsOnExitDistance(beep, (int)px, (int)py);

        if (traceOn){
            // On marque la case actuelle
            int tx = (int)px;
            int ty = (int)py;

            if (tx >= 0 && tx < MAP_W && ty >= 0 && ty < MAP_H) {
                TRACES[ty][tx] = 2.0f;
            }
        }
        else {
            for (int y=0; y<MAP_W; y++)
                for (int x=0; x<MAP_H; x++)
                    if (TRACES[y][x] == 2) TRACES[y][x] = 0;
        }

        KeysAndJoypadHandler(&angle, &px, &py, dt);
        
        // ----- Rendu -----
        BeginDrawing();

        float res[2] = {renderW, renderH};
        SetShaderValue(shader, locWallData,         (int[1]){4},                 SHADER_UNIFORM_INT);
        SetShaderValue(shader, locResolution, res,                               SHADER_UNIFORM_VEC2);
        SetShaderValueV(shader, locLightPos,    lightPositions,                  SHADER_UNIFORM_VEC3,  NUM_LIGHTS);
        SetShaderValueV(shader, locLightColor,  lightColors,                     SHADER_UNIFORM_VEC3,  NUM_LIGHTS);
        SetShaderValueV(shader, locLightRadius, lightRadii,                      SHADER_UNIFORM_FLOAT, NUM_LIGHTS);
        SetShaderValue (shader, locNumLights,   &numLights,                      SHADER_UNIFORM_INT);
        SetShaderValue(shader, locHorizon,       (float[1]){5},                  SHADER_UNIFORM_FLOAT);
        SetShaderValue(shader, locParallaxScale, &parallaxScale,                 SHADER_UNIFORM_FLOAT);
        SetShaderValue(shader, locMinSamples,    (int[1]){PARALLAX_STEPS_MIN},   SHADER_UNIFORM_INT);
        SetShaderValue(shader, locMaxSamples,    (int[1]){PARALLAX_STEPS_MAX},   SHADER_UNIFORM_INT);
        float n = 0.2f;
        SetShaderValue(shader, locNormalStrength, &n,                            SHADER_UNIFORM_FLOAT);
        float playerPos[2]={px,py};        
        SetShaderValue(shader,locPlayer,playerPos,                               SHADER_UNIFORM_VEC2);        
        
        int horizon = halfRenderH + (int)pitch;
        UpdateTexture(ctx.wallDataTex, RenderWallPixelShader(&ctx, px, py, angle, rayOffset,horizon));

#if FXAA
        apply_fxaa(ctx.framebuffer, ctx.tmpFxaa, renderW, renderH);
#endif
        BeginBlendMode(BLEND_ALPHA);
        RenderFloorCeil(&ctx, px, py, angle, horizon);        

        // 2. Upload le framebuffer comme texture de fond
        UpdateTexture(ctx.screenTex, ctx.framebuffer);
        
        // 3. Dessine le fond CPU
        DrawTexturePro(
            ctx.screenTex,
            (Rectangle){0, 0, renderW, renderH},
            (Rectangle){0, 0, SCREEN_W, SCREEN_H},
            (Vector2){0, 0}, 0.0f, WHITE
        );

        BeginShaderMode(shader);

        SetShaderValueTexture(shader, locDiffuse,  ctx.texDiffuse);
        SetShaderValueTexture(shader, locNormal,   ctx.texNormal);
        SetShaderValueTexture(shader, locHeight,   ctx.texHeight);
        SetShaderValueTexture(shader, locWallData, ctx.wallDataTex);
        
        DrawTexturePro(
            ctx.texDiffuse,
            (Rectangle){0,0,ctx.texDiffuse.width,ctx.texDiffuse.height},
            (Rectangle){0,0,SCREEN_W,SCREEN_H},
            (Vector2){0,0},
            0,
            WHITE
        );     

        EndShaderMode();

        RenderSprites(&ctx, px, py, angle);
        // 2. Upload le framebuffersprites comme texture de sprites
        UpdateTexture(ctx.screenTex, ctx.framebuffersprites);
        
        // 3. Dessine les sprites
        DrawTexturePro(
            ctx.screenTex,
            (Rectangle){0, 0, renderW, renderH},
            (Rectangle){0, 0, SCREEN_W, SCREEN_H},
            (Vector2){0, 0}, 0.0f, WHITE
        );
        EndBlendMode();

        if (afficherMap)
            draw_minimap(px, py, angle);

        // ----- Victoire -----
        if (MAP[(int)py][(int)px] == 2)
            gameState = STATE_WIN;

        GState(gameState, &px, &py, tada, siren);

        // ----- HUD -----
        DrawFPS(SCREEN_W - 80, 10);

        char timerBuf[32];
        int  s = (int)gameTimer;
        sprintf(timerBuf, "%02d:%02d", s / 60, s % 60);
        DrawText(timerBuf, SCREEN_W/2 - 30, 10, 24, WHITE);

        DrawText("ZQSD / Fleches : se deplacer", 10, SCREEN_H - 25, 14, GRAY);

        // ----- HUD Pulse -----
        if (pulseRadius > 0.0f)
        {
            // Distance normalisée 0→1 pour la couleur
            //float distExit = sqrtf(powf(exitX - (int)px, 2) + powf(exitY - (int)py, 2));
            float proximity = 1.0f - fminf(distExit / 20.0f, 1.0f);  // 1 = très proche

            // Couleur : vert loin → rouge proche
            Color pulseColor = {
                (unsigned char)(proximity * 255),
                (unsigned char)((1.0f - proximity) * 255),
                0,
                (unsigned char)(pulseRadius / 20.0f * 200)  // fade out
            };

            int cx = SCREEN_W/2;
            int cy = 60;
            DrawCircle(cx, cy, (int)pulseRadius, pulseColor);
            DrawCircleLines(cx, cy, (int)pulseRadius, WHITE);

            // Décroissance
            pulseRadius -= 60.0f * GetFrameTime();
            if (pulseRadius < 0.0f) pulseRadius = 0.0f;
        }

        DrawText( TextFormat( "%dx%d / render %dx%d",  SCREEN_W, SCREEN_H, renderW, renderH), 10,  40,  20, YELLOW);
        
        EndDrawing();
    }

    UnloadSound(beep);
    UnloadSound(tada);
    UnloadSound(siren);
    CloseAudioDevice();
    free(wallDataBuf);
    free(framebuffer);
    free(framebuffersprites);
    free(zBuffer);
    free(tmpFxaa);
    CloseWindow();
    return 0;
}
#pragma endregion