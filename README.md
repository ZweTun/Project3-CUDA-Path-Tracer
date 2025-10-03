# University of Pennsylvania, CIS 5650: GPU Programming and Architecture
## Project 3 - CUDA Path Tracer

* Zwe Tun
  * LinkedIn: https://www.linkedin.com/in/zwe-tun-6b7191256/
* Tested on: Intel(R) i7-14700HX, 2100 Mhz, RTX 5060 Laptop

## Overview 
Path tracing is a rendering technique that produces realistic images by simulating how light travels through a scene. Instead of just tracing rays directly to light sources, it sends rays from the camera that bounce around surfaces, picking up color and brightness based on the material they hit. The downside is that the images start out noisy since the method relies on randomness, but with enough samples the noise fades and the image converges to a natural, physically accurate result.


## Resources 
https://www.cg.tuwien.ac.at/sites/default/files/course/4411/attachments/04_path_tracing.pdf
https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation.html
https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting
https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
