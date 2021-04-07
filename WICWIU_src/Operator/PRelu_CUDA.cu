#ifdef __CUDNN__

#include "PRelu.hpp"

// template class PRelu<int>;
template class PRelu<float>;
// template class PRelu<double>;

/*!
@class PRelu cuda
*/


__global__ void ForwardPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, int weightDim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
          if(pDevInput[idx] > 0.f)
                pDevOutput[idx] = pDevInput[idx];
          else
                pDevOutput[idx] = pDevWeight[idx]* pDevInput[idx];
    }
}


template<typename DTYPE> int PRelu<DTYPE>::ForwardPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        int m_parameterDim = this->GetResult()->GetCapacity();

        DTYPE *m_pDevInput  = input->GetGPUData(pTime);
        DTYPE *m_pDevWeight  = weight->GetGPUData(pTime);
        DTYPE *m_pDevOutput = result->GetGPUData(pTime);

        ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput,  m_parameterDim);

        return TRUE;
}


__global__ void BackPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, float *pDevDelta, float *pDevInputDelta, float *pDevWeightDelta, int weightDim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
          if(pDevOutput[idx] > 0.f){
                pDevInputDelta[idx] += pDevDelta[idx];
                pDevWeightDelta[idx] += 0;
          }
          else{
                pDevInputDelta[idx] += pDevWeight[idx]* pDevDelta[idx];
                pDevWeightDelta[idx] += pDevInput[idx]* pDevDelta[idx];
          }
    }
}


template<typename DTYPE> int PRelu<DTYPE>::BackPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weight_delta = this->GetInput()[1]->GetDelta();
        int m_parameterDim = this->GetResult()->GetCapacity();

        DTYPE *m_pDevInput = input->GetGPUData(pTime);
        DTYPE *m_pDevWeight  = weight->GetGPUData(pTime);
        DTYPE *m_pDevOutput = result->GetGPUData(pTime);

        DTYPE *m_pDevDelta      = this_delta->GetGPUData(pTime);
        DTYPE *m_pDevInputDelta = input_delta->GetGPUData(pTime);
        DTYPE *m_pDevWeightDelta = weight_delta->GetGPUData(pTime);

        BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput, m_pDevDelta, m_pDevInputDelta, m_pDevWeightDelta, m_parameterDim);

        return TRUE;
}

#endif  // ifdef __CUDNN__
