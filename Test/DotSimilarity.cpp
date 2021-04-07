#include "../WICWIU_src/NeuralNetwork.hpp"



//pKey - 모든 Encoder의 hidden 값!
//pQuery - t 시점의 decoder hidden, recurent의 result를 가져올거임!


int main(int argc, char const *argv[]) {


    int EncoderTimeSize = 4;
    int DecoderTimeSize = 6;
    int HiddenSize = 5;
    int BatchSize = 2;

    Tensorholder<float> *pKey = new Tensorholder<float>(Tensor<float>::Random_normal(EncoderTimeSize, BatchSize, 1, 1, HiddenSize, 0.0, 0.1), "key");
    Tensorholder<float> *pQuery = new Tensorholder<float>(Tensor<float>::Random_normal(DecoderTimeSize, BatchSize, 1, 1, HiddenSize, 0.0, 0.1), "query");


    // (*(input0->GetResult()))[0] = 0.1;

    std::cout<<"pKey - 모든 Encoder의 hidden 값!"<<'\n';
    std::cout<<"pQuery - t 시점의 decoder hidden, recurent의 result를 가져올거임!"<<'\n';


    std::cout<<"pKey"<<'\n';
    std::cout << pKey->GetResult()->GetShape() << '\n';
    std::cout << pKey->GetResult() << '\n';

    std::cout<<"pQuery"<<'\n';
    std::cout << pQuery->GetResult()->GetShape() << '\n';
    std::cout << pQuery->GetResult() << '\n';

    Operator<float> *matmul = new DotSimilarity<float>(pKey, pQuery, "matmultest");
    //Operator<float> *matmul = new MatMul<float>(input0, pWeight, "matmultest");


    #ifdef __CUDNN__
          matmul->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          matmul->ForwardPropagate(0);
    #endif

      std::cout << "***********************ForwardPropagate time=0 후****************" << '\n';
      std::cout << matmul->GetResult()->GetShape() << '\n';
      std::cout << matmul->GetResult() << '\n';


      std::cout << "***********************모든 time ForwardPropagate****************" << '\n';

      matmul->ForwardPropagate(1);
      matmul->ForwardPropagate(2);
      matmul->ForwardPropagate(3);
      matmul->ForwardPropagate(4);
      matmul->ForwardPropagate(5);

      std::cout << matmul->GetResult()->GetShape() << '\n';
      std::cout << matmul->GetResult() << '\n';


      matmul->BackPropagate(5);
      matmul->BackPropagate(4);
      matmul->BackPropagate(3);
      matmul->BackPropagate(2);
      matmul->BackPropagate(1);
      matmul->BackPropagate(0);



      // (*(matmul->GetDelta()))[0] = 1;
      // (*(matmul->GetDelta()))[1] = 2;
      // (*(matmul->GetDelta()))[2] = 3;
      // (*(matmul->GetDelta()))[3] = 4;
      //
      // std::cout<<"matmul의 gradient값"<<'\n';
      // std::cout << matmul->GetGradient()->GetShape() << '\n';
      // std::cout << matmul->GetGradient() << '\n';
      //
      // //matmul->BackPropagate(1);
      //
      // std::cout<<"==========================backpropagate 1 이후=========================="<<'\n';
      //
      // std::cout << pWeight->GetGradient()->GetShape() << '\n';
      // std::cout << pWeight->GetGradient() << '\n';
      //
      // matmul->BackPropagate(0);
      //
      // std::cout<<"==========================backpropagate 0 이후=========================="<<'\n';
      //
      // std::cout << pWeight->GetGradient()->GetShape() << '\n';
      // std::cout << pWeight->GetGradient() << '\n';



    }
