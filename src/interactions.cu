#include "interactions.h"

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

__host__ __device__ float schlicksApprox(float cosTheta, float n1, float n2) {
    // Schlick F0 for dielectrics: ((n1 - n2)/(n1 + n2))^2
    float base = (n1 - n2) / (n1 + n2);
    float R0 = base * base;
    float cosThetaClamped = glm::clamp(cosTheta, 0.0f, 1.0f);
    return R0 + (1.0f - R0) * powf(1.0f - cosThetaClamped, 5.0f);
}

__host__ __device__ void refractRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m, thrust::default_random_engine& rng)
{   
    // normalize inputs
    glm::vec3 I = glm::normalize(pathSegment.ray.direction); 
    glm::vec3 N = glm::normalize(normal);

    float cosI = glm::dot(-I, N); 

    float n1 = 1.0f;
    float n2 = m.indexOfRefraction; 
	float eta = n1 / n2; 

	float S = schlicksApprox(cosI, n1, n2); 

    if (S > thrust::uniform_real_distribution<float>(0, 1)(rng)) {
        // reflect
        pathSegment.ray.direction = glm::reflect(I, N);
		pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
		pathSegment.color *= m.color; // reflect color
		pathSegment.color /= S; // attenuate color by reflectance

    }
    else {
        // refract
		pathSegment.ray.direction = glm::refract(I, N, eta);
		pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
		pathSegment.color *= m.color; // refract color
		pathSegment.color /= (1.0f - S); // attenuate color by transmittance
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

	//Depth of field added here.


    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f; 
	pathSegment.color *= m.color;
    

    
}

