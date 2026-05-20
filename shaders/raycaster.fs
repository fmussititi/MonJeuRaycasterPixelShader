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

// Equivalent de ComputeIllumination
vec4 ComputeIllumination(vec2 texSample, vec3 vViewTS, vec2 worldPos, vec2 normal, in float wallType)
{
   vec3 vNormal  = texture(u_normalTexture, texSample).rgb * 2.0 - 1.0;
   vNormal.xy   *= normalStrength;
   vNormal        = normalize(vNormal);

   vec3 vDiffuse = texture(u_diffuseTexture, texSample).rgb;
   vDiffuse = ApplyWallTypeTint(vDiffuse, wallType);
   vec3 outColor = vec3(0.08) * vDiffuse;  // ambient

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
      float spec    = pow(max(0.0, dot(vNormal, vHalf)), 32.0) * 0.3;

      outColor += vDiffuse * NdotL * atten * lightColor[i] * 2.0;
      outColor += vec3(spec) * atten * lightColor[i];
   }   

   return vec4(clamp(outColor, 0.0, 1.0), 1.0);
}


void parallaxOcclusionMapping(in vec2 o_texcoords, in vec3 o_vViewTS, in vec2 o_vParallaxOffsetTS, in vec2 worldPos, in vec2 normal, in float wallType)
{
   vec3 vViewTS   = normalize(o_vViewTS);
   //vec3 vLightTS  = normalize(o_vLightTS);

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
   finalTexCoords = clamp( finalTexCoords,  vec2(0.001),  vec2(0.999) );

   // interpolation of depth values
   float parallaxHeight = curLayerHeight + prevH * weight + nextH * (1.0 - weight);

   // Self-shadow
   //float shadowMultiplier = parallaxSoftShadowMultiplier(
   //   vLightTS, 
   //   finalTexCoords,          // UV après déplacement parallax
   //   parallaxHeight           // hauteur à l'intersection
   //);

   float shadowMultiplier = 1;

   resultingColor = ComputeIllumination(finalTexCoords, vViewTS, worldPos, normal, wallType);
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

   float signedOffsetHorizontal = faceSign * (dot(V, tangent) > 0.0 ? 1.0 : -1.0) * clamp(abs((viewTS.x / max(abs(viewTS.z), 0.1)) * parallaxScale), 0.0, 0.3);
   float offsetVertical = (viewTS.y / viewTS.z) * parallaxScale;

   //vec2 parallaxOffset = vec2(signedOffsetHorizontal, offsetVertical);
   vec2 parallaxOffset = vec2(signedOffsetHorizontal, 0.0);

   parallaxOcclusionMapping(vec2(d.texX, texY), viewTS, parallaxOffset, d.worldPos, d.normal, d.wallType);
   //resultingColor= texture( u_diffuseTexture, vec2(d.texX,texY));
}