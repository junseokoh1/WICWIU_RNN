#include "../WICWIU_src/NeuralNetwork.hpp"



/*
  colab에서 test 만든거랑 동일하도록 만든 test 환경
  seed를 42로 고정시킴!

*/

/*
  batch=1일때하고 batch=2일 때 둘다 pytorch랑 값 동일!
  Forward만 확인해보고
  backward는 확인못해봄!
*/


/*
  GPU, CPU 값을 비교할 수가 없는게....
  weight parameter의 값이 달라짐.....
  GPU에서 다 합쳐서 하나로 넣는데 어떤 순서로 넣는지도 모르겠고....
*/

int main(int argc, char const *argv[]) {

    int time_size = 2;
    int input_size = 2;
    int hidden_size = 3;
    //int output_size = 2;
    int batch_size = 2;

    //Tensor에 UseTime 잘 확인하기!!!
    //처음부터 원하는 값으로 초기화하는 방법은 없나?
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(time_size, batch_size, 1, 1, input_size, 0.0, 0.1), "RecurrentLayer_input");
    Tensorholder<float> *Init_Hidden = new Tensorholder<float>(Tensor<float>::Random_normal(1, batch_size, 1, 1, hidden_size, 0.0, 0.1), "RecurrentLayer_input");

    Tensorholder<float> *pWeight_I2h = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, hidden_size, input_size, 0.0, 0.1), "RecurrentLayer_pWeight_I2h_");
    Tensorholder<float> *pWeight_h2h = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, hidden_size, hidden_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2h_");
    Tensorholder<float> *rBias = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 1, 1, hidden_size, 0.f), "RNN_Bias_");
    //Tensorholder<float> *pWeight_h2o = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, output_size, hidden_size, 0.0, 0.1), "RecurrentLayer_pWeight_h2o_");

#ifdef __CUDNN__
    pWeight_h2h = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, hidden_size, hidden_size+input_size+1, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_");    //bias 1개 일때!!!
#endif  // ifdef __CUDNN__

    if(batch_size == 1){
        //input값 설정 batch = 1
        (*(input0->GetResult()))[0] = 0.3367;
        (*(input0->GetResult()))[1] = 0.1288;
        (*(input0->GetResult()))[2] = 0.2345;
        (*(input0->GetResult()))[3] = 0.2303;

        std::cout << "*****************입력 값****************" << '\n';
        std::cout << input0->GetResult() << '\n';

    }else if(batch_size == 2){

        //input값 설정 batch = 2
        (*(input0->GetResult()))[0] = 0.0562;
        (*(input0->GetResult()))[1] = -0.6382;
        (*(input0->GetResult()))[2] = -1.9187;
        (*(input0->GetResult()))[3] = -0.6441;

        (*(input0->GetResult()))[4] = -0.6061;
        (*(input0->GetResult()))[5] = -0.1425;
        (*(input0->GetResult()))[6] = 0.9727;
        (*(input0->GetResult()))[7] = 2.0038;

        std::cout << "*****************입력 값****************" << '\n';
        std::cout << input0->GetResult() << '\n';

    }


    if(batch_size == 1){

        //Init_hidden값 설정
        (*(Init_Hidden->GetResult()))[0] = -1.1299;
        (*(Init_Hidden->GetResult()))[1] = -0.1863;
        (*(Init_Hidden->GetResult()))[2] = 2.2082;

        std::cout << "*****************init hidden 값****************" << '\n';
        std::cout << Init_Hidden->GetResult() << '\n';

    }else if(batch_size == 2){

        //Init_hidden값 설정
        (*(Init_Hidden->GetResult()))[0] = 0.6622;
        (*(Init_Hidden->GetResult()))[1] = 0.5332;
        (*(Init_Hidden->GetResult()))[2] = 2.7489;

        (*(Init_Hidden->GetResult()))[3] = -0.3841;
        (*(Init_Hidden->GetResult()))[4] = -1.9623;
        (*(Init_Hidden->GetResult()))[5] = -0.3090;

        std::cout << "*****************init hidden 값****************" << '\n';
        std::cout << Init_Hidden->GetResult() << '\n';

    }



    //pWeight_I2h 값 설정
    (*(pWeight_I2h->GetResult()))[0] = 1.3221;
    (*(pWeight_I2h->GetResult()))[1] = 0.8172;
    (*(pWeight_I2h->GetResult()))[2] = -0.7658;
    (*(pWeight_I2h->GetResult()))[3] = -0.7506;
    (*(pWeight_I2h->GetResult()))[4] = 1.3525;
    (*(pWeight_I2h->GetResult()))[5] = 0.6863;


    std::cout << "*****************pWeight_I2h 값****************" << '\n';
    std::cout << pWeight_I2h->GetResult() << '\n';


    //pWeight_h2h값 설정
    (*(pWeight_h2h->GetResult()))[0] = -0.6380;
    (*(pWeight_h2h->GetResult()))[1] = 0.4617;
    (*(pWeight_h2h->GetResult()))[2] = 0.2674;
    (*(pWeight_h2h->GetResult()))[3] = 0.5349;
    (*(pWeight_h2h->GetResult()))[4] = 0.8094;
    (*(pWeight_h2h->GetResult()))[5] = 1.1103;
    (*(pWeight_h2h->GetResult()))[6] = -1.6898;
    (*(pWeight_h2h->GetResult()))[7] = -0.9890;
    (*(pWeight_h2h->GetResult()))[8] = 0.9580;

    std::cout << "*****************pWeight_h2h 값****************" << '\n';
    std::cout << pWeight_h2h->GetResult() << '\n';


    //biad값 설정
    (*(rBias->GetResult()))[0] = -0.3278;
    (*(rBias->GetResult()))[1] = 0.7950;
    (*(rBias->GetResult()))[2] = 0.2815;

    std::cout << "*****************rBias  값****************" << '\n';
    std::cout << rBias->GetResult() << '\n';





    //std::cout << pWeight_x2h->GetResult()->GetShape() << '\n';
    //std::cout << pWeight_x2h->GetResult() << '\n';


    // RNN network 생성
    //Operator<float> *rnn = new SeqRecurrent<float>(input0, pWeight_I2h, pWeight_h2h, rBias, Init_Hidden);     //init hidden 넣은거
    Operator<float> *rnn = new SeqRecurrent<float>(input0, pWeight_I2h, pWeight_h2h, rBias, NULL);            //init hidden 사용X

    //Operator<float> *rnn = new Recurrent<float>(input0, pWeight_I2h, pWeight_h2h, rBias);

    //Operator<float> *rnn = new FastRecurrent<float>(input0, pWeight_I2h, pWeight_h2h, rBias, NULL);         //cpu 성능 개선

    //Operator<float> *rnn = new RecurrentCUDNN2<float>(input0, pWeight_I2h, pWeight_h2h, rBias);

    std::cout << '\n';


    //이렇게 다 setdevice 해주는게 맞아????
    #ifdef __CUDNN__
      std::cout<<"GPU에서 동작 중 입니다!!!"<<'\n';
      cudnnHandle_t m_cudnnHandle;
      cudnnCreate(&m_cudnnHandle);
      //여기서 test할때는 weight들도 setdeviceGPU를 해줘야 가능함!!!
      pWeight_I2h->SetDeviceGPU(m_cudnnHandle, 0);
      pWeight_h2h->SetDeviceGPU(m_cudnnHandle, 0);
      rBias->SetDeviceGPU(m_cudnnHandle, 0);
      //pWeight_h2o->SetDeviceGPU(m_cudnnHandle, 0);
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


    // std::cout<<"GetHidden test!!!"<<'\n';
    //
    // std::cout<<rnn->GetHidden()<<'\n';


    //RNN 최상위에 넘겨줄 delta값 설정
    for(int i = 0; i < time_size * hidden_size * batch_size; i++){
    //  std::cout<<"???"<<'\n';
    //  std::cout<<i;
      (*(rnn->GetDelta()))[i] = 0.3;
    }

    std::cout<<"넘겨주는 gradient 값"<<'\n';
    std::cout<<rnn->GetDelta()->GetShape()<<'\n';
    std::cout<<rnn->GetDelta()<<'\n';

    std::cout << "**********************************************BackPropagate 후*********************************************" << '\n';

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


    std::cout<<"Init_Hidden이 받은 gradient 값"<<'\n';
    std::cout << Init_Hidden->GetDelta()->GetShape() << '\n';
    std::cout << Init_Hidden->GetDelta() << '\n';


    std::cout<<"pWeight_h2h이 받은 gradient 값"<<'\n';
    std::cout<<pWeight_h2h->GetDelta()<<'\n';

    delete input0;
    delete pWeight_I2h;
    delete pWeight_h2h;
    //delete pWeight_h2o;



    }
