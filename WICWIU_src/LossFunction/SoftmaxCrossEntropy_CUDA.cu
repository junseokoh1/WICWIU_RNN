#ifdef __CUDNN__

#include "SoftmaxCrossEntropy.hpp"

// template class SoftmaxCrossEntropy<int>;
template class SoftmaxCrossEntropy<float>;
// template class SoftmaxCrossEntropy<double>;

//작은 값을 선택해주는 inline 함수
/*
inline float min(float x, float y)
{
  return x>y ? y : x ;
}
*/

__global__ void SoftmaxCrossEntropy_ForwardPropagate_kernel(int time, int batchsize, int colsize, float epsilon, float *result, float *label, float *softmaxresult) {
    int result_idx = 0;
    int start      = 0;
    int end        = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batchsize; idx += blockDim.x * gridDim.x) {
        result_idx = time * batchsize + idx;
        start      = result_idx * colsize;
        end        = start + colsize;
        //printf("colsize : %d\n", colsize);
        for (int i = start; i < end; i++) {
            //result[result_idx] += -label[i] * log(MIN(softmaxresult[i], softmaxresult[i] + epsilon));
            result[result_idx] += -label[i] * log(MAX(softmaxresult[i], softmaxresult[i] + epsilon));



            // if(isnan(result[result_idx]) != 0){
            //     printf("nan인 경우 index : %d\n", result_idx);
            //     printf("%f \n", MAX(softmaxresult[i], softmaxresult[i] + epsilon));
            //     printf("%f \n\n", log(MAX(softmaxresult[i], softmaxresult[i] + epsilon)));
            //
            // }


            //result[result_idx] += -label[i] * log(softmaxresult[i]);
        //    printf("\n %d \n", i);
        }
    }
}

__global__ void print_kernel(int time, int batchsize) {
    int result_idx = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batchsize; idx += blockDim.x * gridDim.x) {
        result_idx = time * batchsize + idx;
        printf("\n idx = %d result_idx =%d \n", idx, result_idx);
    }
}

template<typename DTYPE> Tensor<DTYPE> *SoftmaxCrossEntropy<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input         = this->GetTensor();
    Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
    Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
    Tensor<DTYPE> *result        = this->GetResult();

    #ifdef __LOSS__
      std::cout<<"SoftmaxCrossEntropy Forward 호출 time = "<<pTime<<'\n';
      std::cout<<"softmaxcrossentropy 의 입력값 : "<<'\n'<<input<<'\n';
      std::cout<<"softmaxcrossentropy 의 라벨 값 : "<<'\n'<<label<<'\n';
    #endif

    int batchsize = input->GetBatchSize();
    int colsize   = input->GetColSize();

    float alpha = 1.f;
    float beta  = 0.f;

    cudnnTensorDescriptor_t pInputDesc   = input->GetDescriptor();
    cudnnTensorDescriptor_t pSoftMaxDesc = softmaxresult->GetDescriptor();

    DTYPE *pDevInput   = input->GetGPUData(pTime);
    DTYPE *pDevSoftMax = softmaxresult->GetGPUData(pTime);

  //std::cout<<"softmax 실행 전의 결과"<<'\n';
  //std::cout<<softmaxresult<<'\n';

    checkCUDNN(cudnnSoftmaxForward(this->GetCudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &alpha, pInputDesc, pDevInput,
                                   &beta, pSoftMaxDesc, pDevSoftMax));

     // std::cout<<"kernel함수 전의 결과"<<'\n';
     // std::cout<<softmaxresult<<'\n';

    int noBlock = 3, threadsPerBlock = 128;
    GetKernelParameters(batchsize, &noBlock, &threadsPerBlock);

    DTYPE *pDevLabel  = label->GetGPUData(pTime);
    DTYPE *pDevResult = result->GetGPUData(pTime);

//    std::cout<<"softmaxcrossentropy의 계산전 result 값"<<'\n';
//    std::cout<<result<<'\n';

    SoftmaxCrossEntropy_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (0, batchsize, colsize, m_epsilon, pDevResult, pDevLabel, pDevSoftMax);
    //print_kernel<<<noBlock, threadsPerBlock>>>(pTime, batchsize);


    // std::cout<<"softmaxcrossentropy의 계산 결과"<<'\n';
    // std::cout<<result<<'\n';

    return result;
}

__global__ void SoftmaxCrossEntropy_BackPropagate_kernel(int time, int capacity, float *input_delta, float *label, float *softmaxresult) {
    int idx = 0;


    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity; idx += blockDim.x * gridDim.x) {
        idx = time * capacity + idx;

        input_delta[idx] = softmaxresult[idx] - label[idx];
    }
}

template<typename DTYPE> Tensor<DTYPE> *SoftmaxCrossEntropy<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
    Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
    Tensor<DTYPE> *input_delta   = this->GetOperator()->GetDelta();

    int batchsize = input_delta->GetBatchSize();
    int colsize   = input_delta->GetColSize();
    int capacity  = batchsize * colsize;

    DTYPE *pDevSoftMax    = softmaxresult->GetGPUData(pTime);
    DTYPE *pDevLabel      = label->GetGPUData(pTime);
    DTYPE *pDevInputDelta = input_delta->GetGPUData(pTime);

    int noBlock = 3, threadsPerBlock = 128;
    GetKernelParameters(capacity, &noBlock, &threadsPerBlock);

    SoftmaxCrossEntropy_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (0, capacity, pDevInputDelta, pDevLabel, pDevSoftMax);

    return NULL;
}

#endif  // ifdef __CUDNN__
