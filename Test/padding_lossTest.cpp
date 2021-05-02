#include "../WICWIU_src/NeuralNetwork.hpp"



/*

  vocab :

  index 형식
  label = [ [1, 2, 3, 4, 5]
            [?, ?, ?, ?, p]
            [?, ?, p, p, p]

  padding = [5, 4, 2]

*/

int main(int argc, char const *argv[]) {

    int time_size = 5;
    int batch_size = 3;
    int vocab_size = 4;

    srand(time(NULL));

    //Tensor에 UseTime 잘 확인하기!!!
    //처음부터 원하는 값으로 초기화하는 방법은 없나?
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, batch_size, 1, 1, vocab_size, 0.0, 0.1), "loss_input");
    Tensorholder<float> *label = new Tensorholder<float>(Tensor<float>::Zeros(time_size, batch_size, 1, 1, vocab_size), "loss_label");
    Tensorholder<float> *padding = new Tensorholder<float>(Tensor<float>::Zeros(1, 1, 1, 1, batch_size), "loss_label");

    Tensor<float> *labelTensor = label->GetResult();
    Shape *labelShape = label->GetResult()->GetShape();

    (*labelTensor)[Index5D(labelShape, 0, 0, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 1, 0, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 2, 0, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 3, 0, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 4, 0, 0, 0, 0)] = 1;

    (*labelTensor)[Index5D(labelShape, 0, 1, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 1, 1, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 2, 1, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 3, 1, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 4, 1, 0, 0, 0)] = 1;

    (*labelTensor)[Index5D(labelShape, 0, 2, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 1, 2, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 2, 2, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 3, 2, 0, 0, 0)] = 1;
    (*labelTensor)[Index5D(labelShape, 4, 2, 0, 0, 0)] = 1;


    std::cout << "*****************label 값****************" << '\n';
    std::cout << label->GetResult() << '\n';



    (*(padding->GetResult()))[0] = 5;
    (*(padding->GetResult()))[1] = 4;
    (*(padding->GetResult()))[2] = 2;
    // (*(padding->GetResult()))[3] = 2;
    // (*(padding->GetResult()))[4] = 1;

    std::cout << "*****************padding 값****************" << '\n';
    std::cout << padding->GetResult() << '\n';



    //random때문에 값을 확인을 못하겠다...

    LossFunction<float> *SCE_loss = new SoftmaxCrossEntropy_padding<float>(input0, label, padding, "SCE");
    //LossFunction<float> *SCE_loss = new SoftmaxCrossEntropy_padding<float>(input0, label, "SCE");
    //LossFunction<float> *SCE_loss = new SoftmaxCrossEntropy<float>(input0, label, "SCE");


#ifdef __CUDNN__
  std::cout<<"GPU에서 동작 중 입니다!!!"<<'\n';
  cudnnHandle_t m_cudnnHandle;
  cudnnCreate(&m_cudnnHandle);
  //여기서 test할때는 weight들도 setdeviceGPU를 해줘야 가능함!!!
  input0->SetDeviceGPU(m_cudnnHandle, 0);
  label->SetDeviceGPU(m_cudnnHandle, 0);
  padding->SetDeviceGPU(m_cudnnHandle, 0);
  SCE_loss->SetDeviceGPU(m_cudnnHandle, 0);
#endif  // ifdef __CUDNN__

    std::cout << '\n';



    std::cout << "***********************ForwardPropagate time=0 후****************" << '\n';

    #ifdef __CUDNN__
          SCE_loss->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          SCE_loss->ForwardPropagate(0);
    #endif

    std::cout << SCE_loss->GetResult()->GetShape() << '\n';
    std::cout << SCE_loss->GetResult() << '\n';

    #ifdef __CUDNN__
          SCE_loss->ForwardPropagateOnGPU(1);
          SCE_loss->ForwardPropagateOnGPU(2);
          SCE_loss->ForwardPropagateOnGPU(3);
          SCE_loss->ForwardPropagateOnGPU(4);
    #else // ifdef __CUDNN__
          SCE_loss->ForwardPropagate(1);
          SCE_loss->ForwardPropagate(2);
          SCE_loss->ForwardPropagate(3);
          SCE_loss->ForwardPropagate(4);
    #endif

    std::cout<<"SCE 최종 forward 결과 값"<<'\n';
    std::cout << SCE_loss->GetResult()->GetShape() << '\n';
    std::cout << SCE_loss->GetResult() << '\n';


    std::cout << "**********************************************BackPropagate 후*********************************************" << '\n';



    #ifdef __CUDNN__
          SCE_loss->BackPropagateOnGPU(4);
          SCE_loss->BackPropagateOnGPU(3);
          SCE_loss->BackPropagateOnGPU(2);
          SCE_loss->BackPropagateOnGPU(1);
          SCE_loss->BackPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          SCE_loss->BackPropagate(4);
          SCE_loss->BackPropagate(3);
          SCE_loss->BackPropagate(2);
          SCE_loss->BackPropagate(1);
          SCE_loss->BackPropagate(0);
    #endif


    //input의 delta 값을 찍어봐야 확인 가능하지!
    std::cout<<"input이 받은 gradient 값"<<'\n';
    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';


    delete input0;
    delete label;
    delete padding;



    }
