#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "cmath"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        } 

        #define blockSize 128
        int* obuffer;
        int* ibuffer;


        //Shift right for exclusive scan
        __global__ void shiftRight(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            if (index == 0) {
                odata[0] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
		}

        __global__ void kernNaiveScan(int n, int* odata, const int* idata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
        
            if (index >= n) {
                return;
            }

          
            if (index >= offset) {
                odata[index] = idata[(index - offset)] + idata[index];
               
            }
            else {
                odata[index] = idata[index];
            }
          

            

 
 
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
        
            cudaMalloc((void**)&obuffer, n * sizeof(int));
            cudaMalloc((void**)&ibuffer, n * sizeof(int));
            cudaMemcpy(obuffer, odata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(ibuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int numBlocks = (n + blockSize - 1) / blockSize;
            dim3 fullBlocksPerGrid(numBlocks);
      
        
      

            int count = 0;
            for (int layer = 1; layer <= ilog2ceil(n); layer++) {
         
				int offset = powf(2, layer - 1);
               
                kernNaiveScan<<<numBlocks, blockSize>>>(n, obuffer, ibuffer, offset);
       
                cudaDeviceSynchronize();
                count++;
				int* temp = ibuffer;
                ibuffer = obuffer;
				obuffer = temp;


             
          
            }
            shiftRight << <numBlocks, blockSize >> > (n, obuffer, ibuffer);
            cudaDeviceSynchronize();

          
            cudaMemcpy(odata, obuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(obuffer);
            cudaFree(ibuffer);
         
			odata[0] = 0;
        

            timer().endGpuTimer();
     
        }
    }
}
