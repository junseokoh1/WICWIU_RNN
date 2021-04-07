#include "../WICWIU_src/NeuralNetwork.hpp"


//batch 적용해서 실험하기!!!
//batch = 3
//col = 5  -> embedding dim = 5
//row = 4인 이유 : positive, neg, neg, neg

//결과 : 1 3 1 1 4
//pos neg neg neg

//Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput

int main(int argc, char const *argv[]) {
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 3, 1, 1, 5, 0.0, 0.1), "x");
    Tensorholder<float> *pWeight = new Tensorholder<float>(Tensor<float>::Random_normal(1, 3, 1, 4, 5, 0.0, 0.1), "weight");

    Tensorholder<float> *pWeight2 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 10, 5, 0.0, 0.1), "weight");

    std::cout<<pWeight2->GetResult()<<'\n';

     (*(input0->GetResult()))[0] = -0.01339;   (*(input0->GetResult()))[1] = -0.005372;
     (*(input0->GetResult()))[2] = 0.01014;    (*(input0->GetResult()))[3] = 0.08042;
     (*(input0->GetResult()))[4] = 0.09904;

     (*(input0->GetResult()))[5] = 0.0474;     (*(input0->GetResult()))[6] = -0.08407;
     (*(input0->GetResult()))[7] = -0.1686;    (*(input0->GetResult()))[8] = 0.01566;
     (*(input0->GetResult()))[9] = -0.1364;

     (*(input0->GetResult()))[10] = -0.1621;   (*(input0->GetResult()))[11] = -0.01926;
     (*(input0->GetResult()))[12] = 0.1015;    (*(input0->GetResult()))[13] = -0.04208;
     (*(input0->GetResult()))[14] = 0.04131;

     //weight값 설정해주고 test! 60개, 한 batch당 20개
     (*(pWeight->GetResult()))[0] = -0.03788;    (*(pWeight->GetResult()))[1] = -0.1025;    (*(pWeight->GetResult()))[2] = -0.06175;   (*(pWeight->GetResult()))[3] = -0.1139;   (*(pWeight->GetResult()))[4] = -0.03183;
     (*(pWeight->GetResult()))[5] = -0.01698;    (*(pWeight->GetResult()))[6] = -0.153;    (*(pWeight->GetResult()))[7] = 0.0531;   (*(pWeight->GetResult()))[8] = 0.01347;   (*(pWeight->GetResult()))[9] = 0.03892;
     (*(pWeight->GetResult()))[10] = 0.08736;   (*(pWeight->GetResult()))[11] = 0.1212;   (*(pWeight->GetResult()))[12] = 0.006172;  (*(pWeight->GetResult()))[13] = -0.1042;  (*(pWeight->GetResult()))[14] = -0.1368;
     (*(pWeight->GetResult()))[15] = 0.08578;   (*(pWeight->GetResult()))[16] = 0.2073;   (*(pWeight->GetResult()))[17] = 0.09315;  (*(pWeight->GetResult()))[18] = 0.1227;  (*(pWeight->GetResult()))[19] = -0.01955;


     (*(pWeight->GetResult()))[20] = -0.02089;    (*(pWeight->GetResult()))[21] = 0.1231;    (*(pWeight->GetResult()))[22] = -0.06875;   (*(pWeight->GetResult()))[23] = -0.006186;   (*(pWeight->GetResult()))[24] = 0.01286;
     (*(pWeight->GetResult()))[25] = -0.05449;    (*(pWeight->GetResult()))[26] = 0.05843;   (*(pWeight->GetResult()))[27] = 0.01405;   (*(pWeight->GetResult()))[28] = 0.04691; (*(pWeight->GetResult()))[29] = -0.1469;
     (*(pWeight->GetResult()))[30] = 0.006045;   (*(pWeight->GetResult()))[31] = 0.03382;  (*(pWeight->GetResult()))[32] = -0.03095;  (*(pWeight->GetResult()))[33] = -0.00622;  (*(pWeight->GetResult()))[34] = -0.05395;
     (*(pWeight->GetResult()))[35] = -0.07982;   (*(pWeight->GetResult()))[36] = -0.1226;  (*(pWeight->GetResult()))[37] = 0.003088;  (*(pWeight->GetResult()))[38] = 0.05011;  (*(pWeight->GetResult()))[39] = -0.0666;


     (*(pWeight->GetResult()))[40] = 0.1533;    (*(pWeight->GetResult()))[41] = 0.1992;   (*(pWeight->GetResult()))[42] = 0.118;   (*(pWeight->GetResult()))[43] = -0.08967; (*(pWeight->GetResult()))[44] = -0.04522;
     (*(pWeight->GetResult()))[45] = -0.1241;    (*(pWeight->GetResult()))[46] = 0.05166;   (*(pWeight->GetResult()))[47] = -0.03945;   (*(pWeight->GetResult()))[48] = -0.1026; (*(pWeight->GetResult()))[49] = -0.106;
     (*(pWeight->GetResult()))[50] = -0.007063;   (*(pWeight->GetResult()))[51] = 0.05458;  (*(pWeight->GetResult()))[52] = -0.05428;  (*(pWeight->GetResult()))[53] = 0.001459; (*(pWeight->GetResult()))[54] = 0.1831;
     (*(pWeight->GetResult()))[55] = 0.02389;   (*(pWeight->GetResult()))[56] = 0.1054;  (*(pWeight->GetResult()))[57] = 0.03122;  (*(pWeight->GetResult()))[58] = -0.04384; (*(pWeight->GetResult()))[59] = -0.01404;




    std::cout<<"input0"<<'\n';
    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    std::cout<<"weight"<<'\n';
    std::cout << pWeight->GetResult()->GetShape() << '\n';
    std::cout << pWeight->GetResult() << '\n';

    Operator<float> *dotProduct = new DotProduct<float>(pWeight, input0, "dotProducttest");



    #ifdef __CUDNN__
          dotProduct->ForwardPropagateOnGPU(0);
    #else // ifdef __CUDNN__
          dotProduct->ForwardPropagate(0);
    #endif

      std::cout<<"forwardPropagate 결과"<<'\n';
      std::cout << dotProduct->GetResult()->GetShape() << '\n';
      std::cout << dotProduct->GetResult() << '\n';

//dotProduct result, gradient shape
//[1, 3, 1, 1, 4]

      for(int i = 0; i < 12; i++){
        (*(dotProduct->GetDelta()))[i] = 1;
      }

      //(*(dotProduct->GetDelta()))[0] = 2;
      // (*(dotProduct->GetDelta()))[1] = 2;
      // (*(dotProduct->GetDelta()))[2] = 3;
      // (*(dotProduct->GetDelta()))[3] = 4;

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





    }
