#include "../WICWIU_src/NeuralNetwork.hpp"

int main(int argc, char const *argv[]) {

    int vocabsize = 10;
    int skipgramDim = 5;            //embedding dim!

    Tensorholder<float> *input0  = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 4, 0.0, 0.1), "label");

    Tensorholder<float> *pWeight_in = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, vocabsize, skipgramDim, 0.0, 0.01), "SKIPGRAMLayer_pWeight_in_");
    Tensorholder<float> *pWeight_out = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, vocabsize, skipgramDim, 0.0, 0.01), "CBOWLayer_pWeight_out_");


    (*(input0->GetResult()))[0] = 1;
    (*(input0->GetResult()))[1] = 6;
    (*(input0->GetResult()))[2] = 8;
    (*(input0->GetResult()))[3] = 9;

    // (*(pWeight->GetResult()))[0] = 1;
    // (*(pWeight->GetResult()))[1] = 2;
    // (*(pWeight->GetResult()))[2] = 2;
    // (*(pWeight->GetResult()))[3] = 4;
    // (*(pWeight->GetResult()))[4] = 5;
    // (*(pWeight->GetResult()))[5] = 6;

    std::cout<<"pWeight_in"<<'\n';
    std::cout << pWeight_in->GetResult()->GetShape() << '\n';
    std::cout << pWeight_in->GetResult() << '\n';

    std::cout<<"pWeight_out"<<'\n';
    std::cout << pWeight_out->GetResult()->GetShape() << '\n';
    std::cout << pWeight_out->GetResult() << '\n';

    std::cout<<"input0"<<'\n';
    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    Operator<float> *skipgram = new SkipGram<float>(input0, pWeight_in, pWeight_out, "SKIPGRAM_Layer");

    #ifdef __CUDNN__
      std::cout<<"GPU에서 동작 중 입니다."<<'\n';
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      pWeight->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      skipgram->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__

    std::cout<<"==========================ForwardPropagate 시작=========================="<<'\n';

    #ifdef __CUDNN__
          skipgram->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          skipgram->ForwardPropagate(0);
    #endif

      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << skipgram->GetResult()->GetShape() << '\n';
      std::cout << skipgram->GetResult() << '\n';


      //Gradient 값 넣어주기!  shape = 1 X 3
      (*(skipgram->GetGradient()))[0] = 1;
      (*(skipgram->GetGradient()))[1] = 2;
      (*(skipgram->GetGradient()))[2] = 3;



      std::cout<<"skipgram의 gradient값"<<'\n';
      std::cout << skipgram->GetGradient()->GetShape() << '\n';
      std::cout << skipgram->GetGradient() << '\n';




      std::cout<<"==========================backpropagate이후=========================="<<'\n';
      skipgram->BackPropagate(0);
      std::cout <<"pWeight_in gradient값"<<'\n';
      std::cout << pWeight_in->GetGradient()->GetShape() << '\n';
      std::cout << pWeight_in->GetGradient() << '\n';

      std::cout <<"pWeight_out gradient값"<<'\n';
      std::cout << pWeight_out->GetGradient()->GetShape() << '\n';
      std::cout << pWeight_out->GetGradient() << '\n';


    }
