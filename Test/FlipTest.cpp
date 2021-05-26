#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(5, 2, 1, 1, 3, 0.0, 0.1), "x");

    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    Operator<float> *flip = new FlipTimeWise<float>(input0);

  #ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
    cudnnCreate(&m_cudnnHandle);
    input0->SetDeviceGPU(m_cudnnHandle, 0);
    input1->SetDeviceGPU(m_cudnnHandle, 0);
    concat->SetDeviceGPU(m_cudnnHandle, 0);
  #endif  // ifdef __CUDNN__

    std::cout << flip->GetResult()->GetShape() << '\n';
    std::cout << flip->GetResult() << '\n';

#ifdef __CUDNN__
    flip->ForwardPropagateOnGPU();
#else // ifdef __CUDNN__
    flip->ForwardPropagate();
#endif  // ifdef __CUDNN__

    std::cout<<"Forward 결과"<<'\n';
    std::cout << flip->GetResult()->GetShape() << '\n';
    std::cout << flip->GetResult() << '\n';

    for(int i = 0; i < 5 * 1 * 3; i++){
      (*(flip->GetDelta()))[i] = i;
    }

    std::cout<<"this Gardient"<<'\n';
    std::cout << flip->GetDelta()->GetShape() << '\n';
    std::cout << flip->GetDelta() << '\n';

    std::cout<<"backward 호출 전"<<'\n';
    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';

  #ifdef __CUDNN__
      flip->BackPropagateOnGPU();
  #else // ifdef __CUDNN__
      flip->BackPropagate();
  #endif  // ifdef __CUDNN__

    std::cout<<"backward 호출 후"<<'\n';
    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';


    delete input0;
    delete flip;

    return 0;
}
