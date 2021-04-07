#include "../WICWIU_src/NeuralNetwork.hpp"

int main(int argc, char const *argv[]) {

    int vocabsize = 10;
    int embeddingDim = 5;

    //batch 없는거!
    //Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "SKIPGRAMLayer_pWeight_in_");
    //Tensorholder<float> *input0  = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 3, 0.0, 0.1), "label");

    Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "SKIPGRAMLayer_pWeight_in_");
    Tensorholder<float> *input0  = new Tensorholder<float>(Tensor<float>::Random_normal(1, 2, 1, 1, 3, 0.0, 0.1), "label");



    //batch1
    (*(input0->GetResult()))[0] = 0;
    (*(input0->GetResult()))[1] = 4;
    (*(input0->GetResult()))[2] = 9;

    //batch2
    (*(input0->GetResult()))[3] = 1;
    (*(input0->GetResult()))[4] = 6;
    (*(input0->GetResult()))[5] = 7;

    // //batch1
    // (*(input0->GetResult()))[6] = 0;
    // (*(input0->GetResult()))[7] = 4;
    // (*(input0->GetResult()))[8] = 9;
    //
    // //batch2
    // (*(input0->GetResult()))[9] = 1;
    // (*(input0->GetResult()))[10] = 6;
    // (*(input0->GetResult()))[11] = 7;

    // (*(pWeight->GetResult()))[0] = 1;
    // (*(pWeight->GetResult()))[1] = 2;
    // (*(pWeight->GetResult()))[2] = 2;
    // (*(pWeight->GetResult()))[3] = 4;
    // (*(pWeight->GetResult()))[4] = 5;
    // (*(pWeight->GetResult()))[5] = 6;

    std::cout<<"weight"<<'\n';
    std::cout << pWeight->GetResult()->GetShape() << '\n';
    std::cout << pWeight->GetResult() << '\n';

    std::cout<<"input0"<<'\n';
    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    Operator<float> *embedding = new Embedding<float>(pWeight, input0, "embeddingtest");

    #ifdef __CUDNN__
      std::cout<<"GPU에서 동작 중 입니다."<<'\n';
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      pWeight->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      embedding->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__


    #ifdef __CUDNN__
          embedding->ForwardPropagateOnGPU(0);
          //embedding->ForwardPropagateOnGPU(1);
    #else // ifdef __CUDNN__
          embedding->ForwardPropagate(0);
    #endif

      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << embedding->GetResult()->GetShape() << '\n';
      std::cout << embedding->GetResult() << '\n';

      //batch1
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i] = 1;
      }
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i+embeddingDim] = 4;
      }
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i+embeddingDim*2] = 9;
      }

      //batch2
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i+embeddingDim*3] = 2;
      }
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i+embeddingDim*4] = 6;
      }
      for(int i = 0; i < embeddingDim; i++){
          (*(embedding->GetGradient()))[i+embeddingDim*5] = 7;
      }


      std::cout<<"embedding의 gradient값"<<'\n';
      std::cout << embedding->GetGradient()->GetShape() << '\n';
      std::cout << embedding->GetGradient() << '\n';


      std::cout<<"==========================backpropagate이후=========================="<<'\n';
#ifdef __CUDNN__
      embedding->BackPropagateOnGPU(0);
#else // ifdef __CUDNN__
      embedding->BackPropagate(0);
#endif
std::cout<<"???"<<'\n';
      std::cout << pWeight->GetGradient()->GetShape() << '\n';
      std::cout << pWeight->GetGradient() << '\n';


    }
