#include "../WICWIU_src/NeuralNetwork.hpp"

int main(int argc, char const *argv[]) {


    int maxTimesize = 7;
    int hiddensize = 5;
    int encoderTime = 10;
    int decoderTime = 7;
    int batchsize = 3;

    //query - decoder     key,value = encoder

    Tensorholder<float> *query = new Tensorholder<float>(Tensor<float>::Random_normal(decoderTime, batchsize, 1, 1, hiddensize, 0.0, 0.1), "x");
    Tensorholder<float> *key = new Tensorholder<float>(Tensor<float>::Random_normal(encoderTime, batchsize, 1, 1, hiddensize, 0.0, 0.1), "weight");
    Tensorholder<float> *value = new Tensorholder<float>(Tensor<float>::Random_normal(encoderTime, batchsize, 1, 1, hiddensize, 0.0, 0.1), "weight");


    std::cout<<"query"<<'\n';
    std::cout << query->GetResult()->GetShape() << '\n';
    std::cout << query->GetResult() << '\n';

    std::cout<<"key"<<'\n';
    std::cout << key->GetResult()->GetShape() << '\n';
    std::cout << key->GetResult() << '\n';

    std::cout<<"value"<<'\n';
    std::cout << value->GetResult()->GetShape() << '\n';
    std::cout << value->GetResult() << '\n';

    Operator<float> *dotAttention = new DotAttention<float>(key, value, maxTimesize, "dotAttentionTest");

    dotAttention->SetQuery(query);


    for(int i=0; i<decoderTime; i++)
      dotAttention->ForwardPropagate(i);


      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << dotAttention->GetResult()->GetShape() << '\n';
      std::cout << dotAttention->GetResult() << '\n';

   for(int i=decoderTime-1; i>0; i--)
      dotAttention->BackPropagate(i);

      std::cout<<"query gradient"<<'\n';
      std::cout << query->GetGradient()->GetShape() << '\n';
      std::cout << query->GetResult() << '\n';

      std::cout<<"key gradient"<<'\n';
      std::cout << key->GetGradient()->GetShape() << '\n';
      std::cout << key->GetResult() << '\n';

      std::cout<<"value gradient"<<'\n';
      std::cout << value->GetGradient()->GetShape() << '\n';
      std::cout << value->GetResult() << '\n';

/*

      for(int i = 0; i < 12; i++){
        (*(dotProduct->GetDelta()))[i] = 1;
      }


      std::cout<<"dotProduct의 gradient값"<<'\n';
      std::cout << dotProduct->GetGradient()->GetShape() << '\n';
      std::cout << dotProduct->GetGradient() << '\n';

      dotProduct->BackPropagate(0);

      std::cout<<"==========================backpropagate 이후=========================="<<'\n';

      std::cout<<"weight의 gradient"<<'\n';
      std::cout << pWeight->GetGradient()->GetShape() << '\n';
      std::cout << pWeight->GetGradient() << '\n';

      std::cout<<'\n';

      std::cout<<"input의 gradient"<<'\n';
      std::cout << input0->GetGradient()->GetShape() << '\n';
      std::cout << input0->GetGradient() << '\n';

*/

    }
