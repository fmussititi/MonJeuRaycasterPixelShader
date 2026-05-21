#version 330

in vec2 fragTexCoord;
out vec4 resultingColor;

uniform sampler2D wallData;

uniform sampler2D u_diffuseTexture;
uniform sampler2D u_normalTexture;
uniform sampler2D u_heightTexture;
uniform sampler2D texture0;

uniform vec2 resolution;

uniform int g_nMinSamples;
uniform int g_nMaxSamples;

uniform float normalStrength;
uniform vec2 tiling;
uniform vec2 playerPos;
uniform float parallaxScale;
uniform float ambientLight;

#define MAX_LIGHTS 3

uniform vec3  lightPos[MAX_LIGHTS];
uniform vec3  lightColor[MAX_LIGHTS];
uniform float lightRadius[MAX_LIGHTS];
uniform int   numLights;

struct WallData
{
    float texX;

    vec2 normal;

    float dist;

    vec2 worldPos;

    float wallType;
    float side;

    float viewTSX;
    float offset;

    float top;
    float bottom;
};

WallData loadWall()
{
    int col = int(fragTexCoord.x * resolution.x);

    vec4 a = texelFetch(wallData, ivec2(col,0),0);
    vec4 b = texelFetch(wallData, ivec2(col,1),0);
    vec4 c = texelFetch(wallData, ivec2(col,2),0);

    WallData d;

    d.texX     = a.r;
    d.normal   = vec2(a.g,a.b);
    d.dist     = a.a;

    d.worldPos = vec2(b.r,b.g);
    d.wallType = b.b;
    d.side     = b.a;

    d.viewTSX  = c.r;
    d.offset   = c.g;
    d.top      = c.b;
    d.bottom   = c.a;

    return d;
}

vec3 ApplyWallTypeTint(vec3 c, float wallType)
{
   // les wallType arrivent en float depuis la texture
   int t = int(wallType + 0.5);

   if(t == 3) c = mix(c, vec3(1.0),                 0.5); // WHITE
   if(t == 4) c = mix(c, vec3(1.0,0.0,0.0),         0.5); // RED
   if(t == 5) c = mix(c, vec3(0.0,1.0,0.0),         0.5); // GREEN
   if(t == 2) c = mix(c, vec3(1.0,0.84,0.0),        0.7); // GOLD

   return c;
}


float parallaxSoftShadowMultiplier(in vec3 L, in vec2 initialTexCoord, 
                                    in float initialHeight, in vec2 parallaxOffset)
{
   float shadowMultiplier = 1.0;

   if (dot(vec3(0, 0, 1), L) > 0.0)
   {
      float numSamplesUnderSurface = 0.0;
      shadowMultiplier = 0.0;

      float numLayers   = mix(float(g_nMaxSamples), float(g_nMinSamples), 
                              abs(dot(vec3(0, 0, 1), L)));
      float layerHeight = initialHeight / numLayers;

      // Step de la lumière proportionnel au parallax offset
      float lightAngle = max(L.z, 0.01);
      
      float shadowScale = max(length(parallaxOffset), parallaxScale * 0.3);

      vec2 texStep = vec2(L.x, -L.y) * shadowScale / (numLayers * lightAngle);

      float currentLayerHeight   = initialHeight - layerHeight;
      vec2  currentTextureCoords = initialTexCoord + texStep;
      float heightFromTexture    = texture(u_heightTexture, currentTextureCoords).r;
      int   stepIndex            = 1;

      for (int i = 0; i < g_nMaxSamples; i++)
      {
         if (currentLayerHeight <= 0.0) break;

         if (heightFromTexture < currentLayerHeight)
         {
               numSamplesUnderSurface += 1.0;
               float newShadow = (currentLayerHeight - heightFromTexture) *
                                 (1.0 - float(stepIndex) / numLayers);
               shadowMultiplier = max(shadowMultiplier, newShadow);
         }

         stepIndex            += 1;
         currentLayerHeight   -= layerHeight;
         currentTextureCoords += texStep;
         heightFromTexture     = texture(u_heightTexture, currentTextureCoords).r;
      }

      if (numSamplesUnderSurface < 1.0)
         shadowMultiplier = 1.0;
      else
         shadowMultiplier = 1.0 - shadowMultiplier;
   }

   return shadowMultiplier;
}


float parallaxSelfShadow(vec3 lightTS, vec2 texCoord, float startHeight, float bumpScale)
{
   if(lightTS.z <= 0.01)
      return 1.0;

   float numSteps = mix(float(g_nMaxSamples), float(g_nMinSamples), lightTS.z);

   float stepSize = 1.0 / numSteps;

   vec2 delta = vec2(lightTS.x,-lightTS.y) * bumpScale /(numSteps * max(lightTS.z,0.05));

   vec2 uv = texCoord;

   float rayHeight = startHeight + stepSize;

   const float shadowBias = 0.03;

   for(int i=0;i<g_nMaxSamples;i++)
   {
      uv += delta;

      float mapHeight = texture(u_heightTexture,uv).r;

      if(mapHeight > rayHeight + shadowBias)
      {
         return 0.0;
      }

      rayHeight += stepSize;

      if(rayHeight > 1.0)
         break;
   }

   return 1.0;
}


// Equivalent de ComputeIllumination
vec4 ComputeIllumination(vec2 texSample, vec3 vViewTS, vec2 worldPos, vec2 normal, in float wallType, in float shadows[MAX_LIGHTS])
{
   vec3 vNormal  = texture(u_normalTexture, texSample).rgb * 2.0 - 1.0;
   vNormal.xy   *= normalStrength;
   vNormal       = normalize(vNormal);

   vec3 vDiffuse = texture(u_diffuseTexture, texSample).rgb;
   vDiffuse = ApplyWallTypeTint(vDiffuse, wallType);

   vec3 outColor = vec3(ambientLight) * vDiffuse;  // ambient

   vec2 tangent = vec2(-normal.y, normal.x);

   for (int i = 0; i < MAX_LIGHTS; i++)
   {
      if (i >= numLights) break;

      vec2 L2 = lightPos[i].xy - worldPos;
      float dist2 = dot(L2, L2);
      float radius = lightRadius[i];
      if (dist2 > radius * radius) continue;

      float dist = sqrt(dist2);

      // Atténuation
      float Kc = 1.0, Kl = 2.0/radius, Kq = 7.0/(radius*radius);
      float atten = 1.0 / (Kc + Kl*dist + Kq*dist2);
      float fade  = 1.0 - dist/radius;
      atten *= max(fade, 0.0);

      // Direction lumière en tangent space
      vec3 vLightTS = normalize(vec3(dot(L2, tangent), 0.0, dot(L2, normal)));
      vLightTS.z    = abs(vLightTS.z);

      float NdotL = max(0.0, dot(vNormal, vLightTS));

      vec3 vViewTSN = normalize(vViewTS);
      vec3 vHalf    = normalize(vLightTS + vViewTSN);

      // Shininess — plus élevé = reflet plus petit et net, plus bas = reflet large et doux
      float shininess = 64.0;  // 32 = brillant, 4-8 = mat/rugueux

      // Puissance du spec — multiplicateur de l'intensité
      float specStrength = 0.3;  // 0.3 = fort, 0.05-0.1 = subtil
      float spec = pow(max(0.0, dot(vNormal, vHalf)), shininess) * specStrength;

      float shadow = pow(shadows[i], 4.0);
      //float shadow = mix(0.45, 1.0, shadows[i]);
      //outColor += vDiffuse * NdotL * atten * lightColor[i] * 2.0 * shadow;
      outColor += vDiffuse * NdotL * atten * lightColor[i] * shadow;
      outColor += vec3(spec) * atten * lightColor[i] * shadow;
   }   

   return vec4(clamp(outColor, 0.0, 1.0), 1.0);
}


void parallaxOcclusionMapping(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS, in vec2 worldPos, in vec2 normal, in float wallType)
{
   vec3 vViewTS   = normalize(o_vViewTS);

   // Nombre de layers adaptatif
   float numLayers = mix(float(g_nMaxSamples), float(g_nMinSamples), 
                        abs(vViewTS.z));  // tangent space, pas world space

   // height of each layer
   float layerHeight = 1.0 / numLayers;
   // current depth of the layer
   float curLayerHeight = 1.0;
   // shift of texture coordinates for each layer
   vec2 dtex = o_vParallaxOffsetTS / numLayers;

   // current texture coordinates
   vec2 currentTextureCoords = o_texcoords;

   // depth from heightmap
   float heightFromTexture = texture(u_heightTexture, currentTextureCoords).r;

   // while point is above the surface
   for (int i = 0; i < g_nMaxSamples; i++)
   {
      if (heightFromTexture >= curLayerHeight) break;

      // to the next layer
      curLayerHeight -= layerHeight;
      // shift of texture coordinates
      currentTextureCoords -= dtex;
      // new depth from heightmap
      heightFromTexture = texture(u_heightTexture, currentTextureCoords).r;
   }

   ///////////////////////////////////////////////////////////

   // previous texture coordinates
   vec2 prevTCoords = currentTextureCoords + dtex;

   // heights for linear interpolation
   float nextH = heightFromTexture - curLayerHeight;
   float prevH = texture(u_heightTexture, prevTCoords).r
                           - curLayerHeight + layerHeight;

   // proportions for linear interpolation
   //float weight = nextH / (nextH - prevH);
   float denom = nextH - prevH;
   float weight = abs(denom) < 0.0001 ? 0.0 : nextH / denom;

   // interpolation of texture coordinates
   vec2 finalTexCoords = prevTCoords * weight + currentTextureCoords * (1.0-weight);
   //finalTexCoords = clamp( finalTexCoords,  vec2(0.001),  vec2(0.999) );

   // interpolation of depth values
   float parallaxHeight = curLayerHeight + prevH * weight + nextH * (1.0 - weight);

   // Self-shadow par lumière
   vec2 tangent = vec2(-normal.y, normal.x);
   float shadows[MAX_LIGHTS];

   for (int i = 0; i < MAX_LIGHTS; i++)
   {
      if (i >= numLights) { shadows[i] = 1.0; continue; }

      vec2 L2 = lightPos[i].xy - worldPos;
      vec3 vLightTS = normalize(vec3(dot(L2, tangent), 0.0, dot(L2, normal)));
      vLightTS.z = abs(vLightTS.z);

      shadows[i] = parallaxSoftShadowMultiplier(vLightTS, finalTexCoords, parallaxHeight, o_vParallaxOffsetTS);
      //shadows[i] = parallaxSelfShadow(vLightTS, finalTexCoords, parallaxHeight, parallaxScale);
      //shadows[i] = 1;
   }

   resultingColor = ComputeIllumination(finalTexCoords, vViewTS, worldPos, normal, wallType, shadows);
}


void parallaxMapping(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS, in vec2 worldPos, in vec2 normal, in float wallType)
{
   vec3 vViewTS   = normalize(o_vViewTS);

   // Sample la height map aux UV de base
   float initialHeight = texture(u_heightTexture, o_texcoords).r;

   // Utilise directement le parallax offset précalculé dans le VS
   // multiplié par la height — plus la surface est haute, plus elle se décale
   vec2 texCoordOffset = o_vParallaxOffsetTS * initialHeight;

   // UV déplacés
   vec2 texSample = o_texcoords + texCoordOffset;


   // Self-shadow par lumière
   float shadows[MAX_LIGHTS];
   for (int i = 0; i < MAX_LIGHTS; i++)
      shadows[i] = 1;

   // Calcul de la couleur finale
   resultingColor = ComputeIllumination(texSample, vViewTS, worldPos, normal, wallType, shadows);
}


void parallaxMappingOffsetLimiting(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS, in vec2 worldPos, in vec2 normal, in float wallType)
{
   vec3 vViewTS = normalize(o_vViewTS);

   // --- Parallax Mapping with Offset Limiting ---
   // Au lieu de diviser par viewTS.z (instable aux angles rasants),
   // on utilise directement viewTS.xy — l'offset est naturellement borné.
   float initialHeight = texture(u_heightTexture, o_texcoords).r;

   // Offset limiting : on ne divise PAS par z, ce qui évite l'explosion
   // aux angles rasants. L'offset est proportionnel à height * xy seulement.
   //vec2 texCoordOffset = parallaxScale * initialHeight * vViewTS.xy;
   vec2 texCoordOffset = initialHeight * o_vParallaxOffsetTS;
   // (pas de / vViewTS.z ici — c'est le "limiting")

   vec2 texSample = o_texcoords + texCoordOffset;

   float shadows[MAX_LIGHTS];
   for (int i = 0; i < MAX_LIGHTS; i++)
      shadows[i] = 1.0;

   resultingColor = ComputeIllumination(texSample, vViewTS, worldPos, normal, wallType, shadows);
}


// -------------------------------------------------------
// Iterative Parallax Mapping with Slope Information
// Premecz 2006 — offset limiting variant
// -------------------------------------------------------
// Pour de bons résultats, u_normalTexture doit stocker la
// hauteur dans son canal alpha (packed normal+height map).
// Si tu n'as pas ça, on fallback sur u_heightTexture.
// -------------------------------------------------------

vec2 iterParallaxMapping(vec2 texCoords, vec3 vViewTS)
{
   // On normalise le vecteur vue (tangent space)
   // PAS de division par z — c'est l'offset limiting de Premecz
   vec3 V = normalize(vViewTS);

   vec2  uv          = texCoords;
   float currentH    = 0.0;   // hauteur du point courant sur le rayon

   const int ITERATIONS = 4;  // 3-4 suffisent selon le paper (270 FPS vs 345 pour 1)

   for (int i = 0; i < ITERATIONS; i++)
   {
      // Lit la normale et la hauteur au point courant
      vec3  N_sample = texture(u_normalTexture, uv).rgb * 2.0 - 1.0;
      float h_map    = texture(u_heightTexture, uv).r;

      float bumpScale = 0.035;
      // SCALE + BIAS appliqués sur la hauteur
      float h = h_map * bumpScale + (bumpScale * -0.5);

      // Nz = composante Z de la normale lue = cosinus de l'angle local
      // Plus la surface est inclinée, plus Nz est petit → offset réduit
      float Nz = N_sample.z;  // dans [0,1] après unpack, ~1 si surface plate

      // Formule de Premecz (eq. itérative, offset limiting = pas de /Vz) :
      // uv[i+1] = uv[i] + (h - currentH) * Nz * V.xy
      uv       += (h - currentH) * Nz * V.xy;
      currentH += (h - currentH) * Nz * V.z;  // mise à jour de la hauteur courante
   }

   return uv;
}

void iterativeParallaxMapping(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS,
                     in vec2 worldPos, in vec2 normal, in float wallType)
{   
   vec2 finalUV = iterParallaxMapping(o_texcoords, o_vViewTS);

   float shadows[MAX_LIGHTS];
   for (int i = 0; i < MAX_LIGHTS; i++)
      shadows[i] = 1.0;

   resultingColor = ComputeIllumination(finalUV, normalize(o_vViewTS), worldPos, normal, wallType, shadows);
}


void steepParallaxMapping(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS, in vec2 worldPos, in vec2 normal, in float wallType)
{
   vec3 vViewTS   = normalize(o_vViewTS);

   // Nombre de layers adaptatif
   float numLayers = mix(float(g_nMaxSamples), float(g_nMinSamples), 
                        abs(vViewTS.z));  // tangent space, pas world space

   float layerHeight        = 1.0 / numLayers;
   float currentLayerHeight = 1.0;

   // Utilise o_vParallaxOffsetTS précalculé dans le VS
   // au lieu de V.xy / V.z / numLayers
   vec2 dtex = -o_vParallaxOffsetTS / numLayers;

   vec2 currentTextureCoords = o_texcoords;
   float heightFromTexture   = texture(u_heightTexture, currentTextureCoords).r;

   // for borné au lieu de while
   for (int i = 0; i < g_nMaxSamples; i++)
   {
      if (heightFromTexture > currentLayerHeight) break;
      currentLayerHeight    -= layerHeight;
      currentTextureCoords  += dtex;
      heightFromTexture      = texture(u_heightTexture, currentTextureCoords).r;
   }

   float parallaxHeight = currentLayerHeight;

   // Self-shadow par lumière
   vec2 tangent = vec2(-normal.y, normal.x);
   float shadows[MAX_LIGHTS];

   for (int i = 0; i < MAX_LIGHTS; i++)
   {
      if (i >= numLights) { shadows[i] = 1.0; continue; }

      vec2 L2 = lightPos[i].xy - worldPos;
      vec3 vLightTS = normalize(vec3(dot(L2, tangent), 0.0, dot(L2, normal)));
      vLightTS.z = abs(vLightTS.z);

      //shadows[i] = parallaxSoftShadowMultiplier(vLightTS, currentTextureCoords, parallaxHeight, o_vParallaxOffsetTS);
      shadows[i] = parallaxSelfShadow(vLightTS, currentTextureCoords, parallaxHeight, parallaxScale);
      //shadows[i] = 1;
   }

   // Calcul de la couleur finale
   resultingColor = ComputeIllumination(currentTextureCoords, vViewTS, worldPos, normal, wallType, shadows);
}


// -------------------------------------------------------
// Contact Refinement Parallax Mapping — Riccardi 2019
// Adapté depuis ParallaxOcclusionMapping.cginc (Unity)
// Erreur : 1/(numLayers²) au lieu de 1/numLayers
// -------------------------------------------------------

// Phase de ray marching générique — utilisée deux fois :
// une fois grossière, une fois fine sur l'intervalle trouvé.
// Retourne les UV à l'intersection.
vec2 raytraceLoop(
   vec2        startUV,
   vec2        deltaUV,        // offset UV par step
   float       deltaH,         // descente en hauteur par step
   int         steps,
   inout float currentH,       // hauteur du rayon (mise à jour)
   inout float mapH)           // hauteur de la heightmap (mise à jour)
{
   vec2 uv = startUV;

   for (int i = 0; i < steps; i++)
   {
      // On est sous la surface → on s'arrête
      if (currentH <= mapH) break;

      uv       -= deltaUV;
      currentH -= deltaH;
      mapH      = texture(u_heightTexture, uv).r;
   }
   return uv;
}


vec2 contactRefinementPOM(vec2 texCoords, vec3 vViewTS, out float rayH)
{
   vec3 V = normalize(vViewTS);

   // Nombre de steps adaptatif selon l'angle
   float numLayers = mix(float(g_nMaxSamples), float(g_nMinSamples), abs(V.z));

   // Delta UV et delta hauteur par step (offset limiting : pas de /z)
   // On utilise parallaxScale comme facteur global
   vec2  deltaUV = (V.xy * parallaxScale) / numLayers;
   float deltaH  = 1.0 / numLayers;

   // ---- Phase 1 : marche grossière ----
   vec2  uv       = texCoords;
   float currentH = 1.0;                        // rayon part du sommet
   float mapH     = texture(u_heightTexture, uv).r;

   uv = raytraceLoop(uv, deltaUV, deltaH, int(numLayers), currentH, mapH);

   // ---- Phase 2 : raffinement sur l'intervalle [step-1, step] ----
   // Recule d'un step — on repasse au-dessus de la surface
   uv       += deltaUV;
   currentH += deltaH;
   mapH      = texture(u_heightTexture, uv).r;

   // Divise la précision par numLayers → erreur finale 1/N²
   float refineDelta = deltaH  / numLayers;
   vec2  refineUV    = deltaUV / numLayers;

   uv = raytraceLoop(uv, refineUV, refineDelta, int(numLayers), currentH, mapH);

   rayH = currentH;
   return uv;
}


void contactRefinementParallaxOcclusionMapping(in vec2  o_texcoords, in vec3  o_vViewTS, in vec2 offset, in vec2 worldPos, in vec2 normal, in float wallType)
{
   float rayH;
   vec2 finalUV = contactRefinementPOM(o_texcoords, o_vViewTS, rayH);

   // Self-shadow (réutilise parallaxSelfShadow existant)
   vec2  tangent = vec2(-normal.y, normal.x);
   float shadows[MAX_LIGHTS];

   for (int i = 0; i < MAX_LIGHTS; i++)
   {
      if (i >= numLights) { shadows[i] = 1.0; continue; }

      vec2 L2       = lightPos[i].xy - worldPos;
      vec3 vLightTS = normalize(vec3(dot(L2, tangent), 0.0, dot(L2, normal)));
      vLightTS.z    = abs(vLightTS.z);

      //float startH = texture(u_heightTexture, finalUV).r;
      //shadows[i] = parallaxSoftShadowMultiplier(vLightTS, finalUV, rayH, offset);
      shadows[i] = parallaxSelfShadow(vLightTS, finalUV, rayH, parallaxScale);
   }

   resultingColor = ComputeIllumination(finalUV, normalize(o_vViewTS), worldPos, normal, wallType, shadows);
}

void main()
{
   WallData d = loadWall();

   float y = fragTexCoord.y * resolution.y;

   if(y<d.top || y>d.bottom)
      discard;

   float wallHeight=d.bottom-d.top;

   if(wallHeight<=0.0)
      wallHeight=1.0;

   vec2 tangent = vec2(-d.normal.y,d.normal.x);

   // direction horizontale du rayon
   vec2 V = normalize(d.worldPos-playerPos);

   float wall_h_real=d.bottom-d.top;

   if(wall_h_real<1.0)
      wall_h_real=1.0;

   // même formule que ton soft
   float screenY = (y-(d.top+wall_h_real*0.5)) / (wall_h_real*0.5);
   float viewY = -screenY;

   // reconstruction UV verticale
   float texY = (y - d.top) / wallHeight;
   texY = clamp(texY, 0.0, 1.0);

   // reconstruction direction caméra 3D
   vec3 viewDir =  normalize( vec3( V.x, viewY, V.y ));

   // projection tangent-space
   vec3 viewTS= vec3( dot(viewDir.xz,tangent), viewDir.y, dot(viewDir.xz,d.normal));

   //vec2 parallaxOffset = vec2(d.offset,0);
  
   float faceSign=1.0f;

   if(d.side==0 && d.normal.x<0) faceSign= 1.0f;
   if(d.side==0 && d.normal.x>0) faceSign=-1.0f;
   if(d.side==1 && d.normal.y<0) faceSign=-1.0f;
   if(d.side==1 && d.normal.y>0) faceSign= 1.0f;

   float localSafeZ = max(abs(viewTS.z), 0.05);
   float signedOffsetHorizontal = faceSign * (viewTS.x / localSafeZ) * parallaxScale;
   float offsetVertical = (viewTS.y / localSafeZ) * parallaxScale;

   vec2 parallaxOffset = vec2(signedOffsetHorizontal, offsetVertical);
   //vec2 parallaxOffset = vec2(signedOffsetHorizontal, 0.0);

   float tileTexX = d.texX * tiling.x;
   float tileTexY = texY * tiling.y;

   //steepParallaxMapping(vec2(tileTexX, tileTexY), viewTS, parallaxOffset, d.worldPos, d.normal, d.wallType);
   //parallaxOcclusionMapping(vec2(tileTexX, tileTexY), viewTS, parallaxOffset, d.worldPos, d.normal, d.wallType);
   contactRefinementParallaxOcclusionMapping(vec2(tileTexX, tileTexY), vec3(faceSign * viewTS.x, viewTS.y, viewTS.z), parallaxOffset, d.worldPos, d.normal, d.wallType);
   //parallaxMapping(vec2(tileTexX, tileTexY), viewTS, parallaxOffset, d.worldPos, d.normal, d.wallType);
   //parallaxMappingOffsetLimiting(vec2(tileTexX, tileTexY), viewTS, vec2(faceSign * viewTS.x * parallaxScale, 0.0), d.worldPos, d.normal, d.wallType);
   //iterativeParallaxMapping(vec2(tileTexX, tileTexY), vec3(faceSign * viewTS.x, viewTS.y, viewTS.z), vec2(0.0, 0.0), d.worldPos, d.normal, d.wallType);
}