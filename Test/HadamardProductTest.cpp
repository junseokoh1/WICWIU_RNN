#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *cellState = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 5, 0.0, 0.1), "x");
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 5, 0.0, 0.1), "label");


    (*(input0->GetResult()))[0] = 1;
    (*(input0->GetResult()))[1] = 2;
    (*(input0->GetResult()))[2] = 3;
    (*(input0->GetResult()))[3] = 4;
    (*(input0->GetResult()))[4] = 5;

    (*(cellState->GetResult()))[0] = 5;
    (*(cellState->GetResult()))[1] = 4;
    (*(cellState->GetResult()))[2] = 3;
    (*(cellState->GetResult()))[3] = 2;
    (*(cellState->GetResult()))[4] = 1;


    std::cout<<"cellState"<<'\n';
    std::cout << cellState->GetResult()->GetShape() << '\n';
    std::cout << cellState->GetResult() << '\n';

    std::cout<<"input0"<<'\n';
    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    //Operator<float> *transpose = new ReShape<float>(cellState, 5, 1, "transpose");
    Operator<float> *matmul = new Hadamard<float>(cellState, input0, "matmultest");

    #ifdef __CUDNN__
      std::cout<<"GPU에서 동작 중 입니다."<<'\n';
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      cellState->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      matmul->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__


    #ifdef __CUDNN__
          matmul->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          matmul->ForwardPropagate(0);
    #endif

      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << matmul->GetResult()->GetShape() << '\n';
      std::cout << matmul->GetResult() << '\n';


      //for(int i = 0; i < 2; i++){
      //  (*(matmul->GetDelta()))[i] = 1;
      //}

      (*(matmul->GetDelta()))[0] = 1;
      (*(matmul->GetDelta()))[1] = 1;
      (*(matmul->GetDelta()))[2] = 1;
      (*(matmul->GetDelta()))[3] = 1;
      (*(matmul->GetDelta()))[4] = 1;

      std::cout<<"matmul의 gradient값"<<'\n';
      std::cout << matmul->GetGradient()->GetShape() << '\n';
      std::cout << matmul->GetGradient() << '\n';

      matmul->BackPropagate(0);

      std::cout<<"==========================backpropagate 이후=========================="<<'\n';

      std::cout<<"cellState의 결과"<<'\n';
      std::cout << cellState->GetGradient()->GetShape() << '\n';
      std::cout << cellState->GetGradient() << '\n';


      std::cout<<"input0의 결과"<<'\n';

      std::cout << input0->GetGradient()->GetShape() << '\n';
      std::cout << input0->GetGradient() << '\n';



    }
