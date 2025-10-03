#include "interactions.h"
#include "intersections.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void directLighting(
    PathSegment& pathSegment,
    glm::vec3 hitPoint,
    glm::vec3 surfNormal,
    const Material& mat,               
    const Geom* lights,                
    const Material* materials,         
    int numberOfLights,
    thrust::default_random_engine& rng, Geom* geoms, int sceneGeomCount)
{
  
    if (numberOfLights <= 0) return;

    // 1) choose a random light 
    float rand = thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    int lightIdx = glm::clamp(int(rand * numberOfLights), 0, numberOfLights - 1);
    const Geom& L = lights[lightIdx];
    const Material& Lmat = materials[L.materialid];           
    glm::vec3 throughput = Lmat.color * Lmat.emittance;               

    // 2) sample a face 
    float rFace = thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float u = thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    float v = thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
    int face = glm::clamp(int(rFace * 6.0f), 0, 5);


    //Cube geometry face select
    glm::vec3 half = glm::abs(L.scale) * 0.5f;
    float xmin = L.translation.x - half.x, xmax = L.translation.x + half.x;
    float ymin = L.translation.y - half.y, ymax = L.translation.y + half.y;
    float zmin = L.translation.z - half.z, zmax = L.translation.z + half.z;

    glm::vec3 pL, nL;
    float area = 0.0f;

    if (face == 0) {
        pL = glm::vec3(xmax, ymin + u * (ymax - ymin), zmin + v * (zmax - zmin));
        nL = glm::vec3(1, 0, 0);
        area = 1.0f / ((ymax - ymin) * (zmax - zmin));
    }
    else if (face == 1) {
        pL = glm::vec3(xmin, ymin + u * (ymax - ymin), zmin + v * (zmax - zmin));
        nL = glm::vec3(-1, 0, 0);
        area = 1.0f / ((ymax - ymin) * (zmax - zmin));
    }
    else if (face == 2) {
        pL = glm::vec3(xmin + u * (xmax - xmin), ymax, zmin + v * (zmax - zmin));
        nL = glm::vec3(0, 1, 0);
        area = 1.0f / ((xmax - xmin) * (zmax - zmin));
    }
    else if (face == 3) {
        pL = glm::vec3(xmin + u * (xmax - xmin), ymin, zmin + v * (zmax - zmin));
        nL = glm::vec3(0, -1, 0);
        area = 1.0f / ((xmax - xmin) * (zmax - zmin));
    }
    else if (face == 4) {
        pL = glm::vec3(xmin + u * (xmax - xmin), ymin + v * (ymax - ymin), zmax);
        nL = glm::vec3(0, 0, 1);
        area = 1.0f / ((xmax - xmin) * (ymax - ymin));
    }
    else { // face == 5 or any other unexpected value
        pL = glm::vec3(xmin + u * (xmax - xmin), ymin + v * (ymax - ymin), zmin);
        nL = glm::vec3(0, 0, -1);
        area = 1.0f / ((xmax - xmin) * (ymax - ymin));
    }

    // 3) shadow ray toward sampled point
    glm::vec3 origin = hitPoint + surfNormal;
    glm::vec3 toLight = pL - origin;
    float dist2 = glm::dot(toLight, toLight);
    float dist = sqrtf(dist2);
    glm::vec3 wi = toLight / dist;

    float cosH = glm::max(0.0f, glm::dot(surfNormal, wi));
    float cosL = glm::max(0.0f, glm::dot(nL, -wi));

    Ray shadow; shadow.origin = origin; shadow.direction = wi;
    glm::vec3 tmpIP, tmpN; bool tmpOutside;

    
    float tHit = boxIntersectionTest(L, shadow, tmpIP, tmpN, tmpOutside); 
    if (tHit > 0.0f && tHit <= dist) {
        float p_sel = 1.0f / float(numberOfLights);
        float p = p_sel * area;

        if (p > 0.0f) {
            glm::vec3 f = mat.color * (1.0f / PI);  
            glm::vec3 contrib = throughput * f * (cosH * cosL) / (dist2 * p);
            pathSegment.accumulated += pathSegment.color * contrib;

        }
    }
    

 
 
}

__host__ __device__ float schlicksApprox(float cosTheta, float n1, float n2) {

    float base = (n1 - n2) / (n1 + n2);
    float R0 = base * base;
    float cos = glm::clamp(cosTheta, 0.0f, 1.0f);
    return R0 + (1.0f - R0) * powf(1.0f - cos, 5.0f);
}

__host__ __device__ void refractRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m, thrust::default_random_engine& rng)
{
    glm::vec3 I = glm::normalize(pathSegment.ray.direction);
    glm::vec3 N = glm::normalize(normal);

 
    float cosI = glm::dot(-I, N);
    float n1 = 1.0f;
    float n2 = m.indexOfRefraction;
    glm::vec3 N2 = N;
    float eta = n1 / n2;

    if (cosI < 0.0f) {
       
        N2 = -N;
        cosI = glm::dot(-I, N2);
        n1 = m.indexOfRefraction;
        n2 = 1.0f;
        eta = n1 / n2; 
    }

    // Schlick's approximation
    float R = schlicksApprox(cosI, n1, n2); 
    R = glm::clamp(R, 0.0f, 1.0f);

    float etaRel = n1 / n2;
    float k = 1.0f - etaRel * etaRel * (1.0f - cosI * cosI);

    const float small = 1e-4f;


    float rand = thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng);

    if (k < 0.0f) {

        pathSegment.ray.direction = glm::reflect(I, N2);
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
        pathSegment.color *= m.color;
        return;
    }
    else {
        // refraction is possible
        glm::vec3 refracted = glm::refract(I, N2, etaRel);

        if (rand < R) {
     
            pathSegment.ray.direction = glm::reflect(I, N2);
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
         
            pathSegment.color *= m.color;

            if (R > small) {
                pathSegment.color /= R;
            }
            else {
				pathSegment.color /= small;
            }
          
        }
        else {

            pathSegment.ray.direction = glm::normalize(refracted);
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
          
            pathSegment.color *= m.color;

            if ((1.0f - R) > small) {
                pathSegment.color /= (1.0f - R);
            }
            else {
				pathSegment.color /= small;
            }
        }
    }
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f; 
	pathSegment.color *= m.color;
    

    
}

