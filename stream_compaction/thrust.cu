#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"


namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* obuffer;
            int* ibuffer;
            cudaMalloc(&obuffer, n * sizeof(int));
            cudaMalloc(&ibuffer, n * sizeof(int));

            cudaMemcpy(ibuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            thrust::device_ptr<const int> dev_in = thrust::device_pointer_cast(ibuffer);
            thrust::device_ptr<int> dev_out = thrust::device_pointer_cast(obuffer);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
            timer().endGpuTimer();

            cudaMemcpy(odata, obuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(obuffer);
            cudaFree(ibuffer);
        }
    }
}
