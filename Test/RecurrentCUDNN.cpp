#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {

    int time_size = 2;
    int input_size = 7;
    int hidden_size = 9;
    int output_size = 2;

    //Tensor에 UseTime 잘 확인하기!!!
    //처음부터 원하는 값으로 초기화하는 방법은 없나?
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, 1, 1, 1, input_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2o_");
    Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, hidden_size, hidden_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2h_");
    Tensorholder<float> *rBias = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, hidden_size, 0.f), "RNN_Bias_");


    //input값 설정
    (*(input0->GetResult()))[0] = 1;
    (*(input0->GetResult()))[1] = 0;
    (*(input0->GetResult()))[2] = 0;
    (*(input0->GetResult()))[3] = 1;

    (*(input0->GetResult()))[4] = 0.4;
    (*(input0->GetResult()))[5] = 0.26;


    std::cout << "*****************입력 값****************" << '\n';
    std::cout << input0->GetResult() << '\n';


    //GPU설정을 위한 weight값
    (*(pWeight_x2h->GetResult()))[1] = -1;
    (*(pWeight_x2h->GetResult()))[0] = 0.4;
    (*(pWeight_x2h->GetResult()))[3] = 0.5;
    (*(pWeight_x2h->GetResult()))[2] = 0.3;

    (*(pWeight_h2h->GetResult()))[1] = 0.3;
    (*(pWeight_h2h->GetResult()))[0] = 0.03;
    (*(pWeight_h2h->GetResult()))[3] = 0.25;
    (*(pWeight_h2h->GetResult()))[2] = 0.2;

    (*(pWeight_h2o->GetResult()))[1] = -0.5;
    (*(pWeight_h2o->GetResult()))[0] = -1.4;
    (*(pWeight_h2o->GetResult()))[3] = 1.3;
    (*(pWeight_h2o->GetResult()))[2] = 0.1;


    (*(pWeight_h2h->GetResult()))[11] = 0.1;
*/

    //std::cout << pWeight_x2h->GetResult()->GetShape() << '\n';
    //std::cout << pWeight_x2h->GetResult() << '\n';

    std::cout << "pweight_h2h 출력"<<'\n';
    std::cout << pWeight_h2h->GetResult()->GetShape() << '\n';
    std::cout << pWeight_h2h->GetResult() << '\n';

    Operator<float> *rnn = new RecurrentCUDNN<float>(input0, pWeight_x2h, pWeight_h2h, rBias, "RNN");

    std::cout << '\n';


    //이렇게 다 setdevice 해주는게 맞아????
    #ifdef __CUDNN__
      std::cout<<"GPU에서 동작 중 입니다!!!"<<'\n';
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      //여기서 test할때는 weight들도 setdeviceGPU를 해줘야 가능함!!!
      pWeight_x2h->SetDeviceGPU(m_cudnnHandle, 0);
      pWeight_h2h->SetDeviceGPU(m_cudnnHandle, 0);
      rBias->SetDeviceGPU(m_cudnnHandle, 0);
      pWeight_h2o->SetDeviceGPU(m_cudnnHandle, 0);
      input0->SetDeviceGPU(m_cudnnHandle, 0);
      rnn->SetDeviceGPU(m_cudnnHandle, 0);
    #endif  // ifdef __CUDNN__


    std::cout << "***********************ForwardPropagate time=0 후****************" << '\n';

    #ifdef __CUDNN__
          rnn->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          rnn->ForwardPropagate(0);
    #endif

    std::cout << rnn->GetResult()->GetShape() << '\n';
    std::cout << rnn->GetResult() << '\n';

    std::cout << "***********************ForwardPropagate time=1 후****************" << '\n';

    #ifdef __CUDNN__
          rnn->ForwardPropagateOnGPU(1);
    #else // ifdef __CUDNN__
          rnn->ForwardPropagate(1);
    #endif

    std::cout<<"RNN의 최종 forward 결과 값"<<'\n';
    std::cout << rnn->GetResult()->GetShape() << '\n';
    std::cout << rnn->GetResult() << '\n';



    //RNN 최상위에 넘겨줄 delta값 설정
    for(int i = 0; i < time_size * hidden_size; i++){
    //  std::cout<<"???"<<'\n';
    //  std::cout<<i;
      (*(rnn->GetDelta()))[i] = 0.3;
    }

    std::cout << "******************************************BackPropagate 후***********************************" << '\n';

    #ifdef __CUDNN__
          rnn->BackPropagateOnGPU(1);
          rnn->BackPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          rnn->BackPropagate(1);
          rnn->BackPropagate(0);
    #endif  // ifdef __CUDNN__

    //input의 delta 값을 찍어봐야 확인 가능하지!
    std::cout<<"input이 받은 gradient 값"<<'\n';
    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';

    //std::cout<<"pWeight_h2h이 받은 gradient 값"<<'\n';




    delete input0;
    delete pWeight_x2h;
    delete pWeight_h2h;
    delete pWeight_h2o;



    }
