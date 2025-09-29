#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        #define blockSize 128
        int* obuffer;
        int* ibuffer;


        __global__ void upSweep(int n, int* idata, int layer) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

			int skip = 1 << (layer + 1); // powf(2, layer + 1);

            int i = (index) * skip;
            if (i + skip - 1 >= n) {
                return;
            }
          
           
            idata[int(i + skip - 1)] += idata[int(i + (skip >> 1) - 1)];

        
        }


        __global__ void downSweep(int n, int* idata, int layer) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
          
   
			int skip = 1 << (layer + 1); // powf(2, layer + 1);
            int i = index * skip;
            if (i + skip - 1 >= n) {
                return;
			}   
          


            int t = idata[int(i + (skip >> 1) - 1)];
            
        
            idata[int(i + (skip >> 1) - 1)] = idata[int(i + skip - 1)];
            idata[int(i + skip - 1)] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
           

            int size = 1 << ilog2ceil(n); 
       
            //Init Buffers 
            int* obuffer;
            int* ibuffer;
            cudaMalloc((void**)&obuffer, size * sizeof(int));
            cudaMalloc((void**)&ibuffer, size * sizeof(int));
            cudaMemset(ibuffer, 0, (size) * sizeof(int));

            cudaMemcpy(obuffer, odata, size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(ibuffer, idata, size * sizeof(int), cudaMemcpyHostToDevice);
          
            cudaDeviceSynchronize();
          
			// up sweep
            int numBlocks;


            for (int layer = 0; layer <= ilog2ceil(size) - 1; layer++) {
                int numThreads = size / int(powf(2, layer + 1));
				numBlocks = (numThreads + blockSize - 1) / blockSize;
                upSweep<<<numBlocks, blockSize>>>(size, ibuffer, layer);
                cudaDeviceSynchronize();
               
             
            }

	        //Set ibuffer[n - 1] = 0 
			cudaMemset(ibuffer + size - 1, 0, sizeof(int));
         
         

           // down sweep 
            for (int layer = ilog2ceil(size) - 1; layer >= 0; layer--) {
                int numThreads = size / int(powf(2, layer + 1));
      

                numBlocks = (numThreads + blockSize - 1) / blockSize;
                downSweep<<<numBlocks, blockSize>>>(size, ibuffer, layer);
                cudaDeviceSynchronize();
               

			}
            timer().endGpuTimer();



   
            cudaMemcpy(odata, ibuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
         
         
		
			cudaFree(ibuffer);
			cudaFree(obuffer);
            
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

          //  timer().startGpuTimer();
			//Init Buffers
            int* boolBufferDevice; 
			int* obuffer;
            int* ibuffer;
			int* scanBuffer = new int[n];
			int* scanBufferDevice;

			cudaMalloc((void**)&scanBufferDevice, n * sizeof(int));
            cudaMalloc((void**)&boolBufferDevice, n * sizeof(int));
            cudaMalloc((void**)&ibuffer, n * sizeof(int));
            cudaMalloc((void**)&obuffer, n * sizeof(int));

            cudaMemcpy(ibuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
           
            int numBlocks = (n + blockSize - 1) / blockSize;
            dim3 fullBlocksPerGrid(numBlocks);
            //Map to boolean 
            Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, boolBufferDevice, ibuffer);
           
			int* boolBuffer = new int[n];
            cudaMemcpy(boolBuffer, boolBufferDevice, n * sizeof(int), cudaMemcpyDeviceToHost);

          
            //Scan boolean array 
            scan(n, scanBuffer, boolBuffer);
   

            cudaMemcpy(scanBufferDevice, scanBuffer, n * sizeof(int), cudaMemcpyHostToDevice);
			
            Common::kernScatter <<<fullBlocksPerGrid, blockSize >> > (n, obuffer, ibuffer, boolBufferDevice, scanBufferDevice);
            ////Scatter results 
       
         
            //  timer().endGpuTimer();

			//Get count by looking at last element of scan + last element of bool
			//Last element of scan holds number of elements up to n-1, so 
            //add bool[n-1] to get the full count
            int count;
            cudaMemcpy(&count, scanBufferDevice + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int lastBool;
            cudaMemcpy(&lastBool, boolBufferDevice + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += lastBool;
            cudaFree(boolBuffer);
            cudaFree(scanBuffer);
            cudaMemcpy(odata, obuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

		
         
			return count;

       
           
        }
    }
}
