#ifndef __SKIPGRAM_LAYER__
#define __SKIPGRAM_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class SKIPGRAMLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief CBOWLayer 클래스 생성자
    @details CBOWLayer 클래스의 Alloc 함수를 호출한다.*/
    SKIPGRAMLayer(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, vocabsize, embeddingDim, pName);
    }

    /*!
    @brief CBOWLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~SKIPGRAMLayer() {}


    int Alloc(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        // std::cout<<"SKIPGRAMLayer vocabsize : "<<vocabsize<<'\n';

        //------------------------------weight 생성-------------------------
        //Win 여기서 window 사이즈만큼 곱하기 안해주는 이유 : input에서 잘라서 값 복사해서 처리해주기???
        Tensorholder<DTYPE> *pWeight_in = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "SKIPGRAMLayer_pWeight_in_" + pName);

        //이건 zero로 설정하더라....
        Tensorholder<DTYPE> *pWeight_out = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "SKIPGRAMLayer_pWeight_out_" + pName);
        //Tensorholder<DTYPE> *pWeight_out = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, 1, 1, vocabsize, embeddingDim), "SKIPGRAMLayer_pWeight_out_" + pName);


        //debugging을 위해서 weight값 설정하기!!!
        //vocab size = 10, embedding = 5로 하자!
/*
        (*(pWeight_in->GetResult()))[0] = -0.03788;    (*(pWeight_in->GetResult()))[1] = -0.1025;    (*(pWeight_in->GetResult()))[2] = -0.06175;   (*(pWeight_in->GetResult()))[3] = -0.1139;   (*(pWeight_in->GetResult()))[4] = -0.03183;
        (*(pWeight_in->GetResult()))[5] = -0.01698;    (*(pWeight_in->GetResult()))[6] = -0.153;    (*(pWeight_in->GetResult()))[7] = 0.0531;   (*(pWeight_in->GetResult()))[8] = 0.01347;   (*(pWeight_in->GetResult()))[9] = 0.03892;
        (*(pWeight_in->GetResult()))[10] = 0.08736;   (*(pWeight_in->GetResult()))[11] = 0.1212;   (*(pWeight_in->GetResult()))[12] = 0.006172;  (*(pWeight_in->GetResult()))[13] = -0.1042;  (*(pWeight_in->GetResult()))[14] = -0.1368;
        (*(pWeight_in->GetResult()))[15] = 0.08578;   (*(pWeight_in->GetResult()))[16] = 0.2073;   (*(pWeight_in->GetResult()))[17] = 0.09315;  (*(pWeight_in->GetResult()))[18] = 0.1227;  (*(pWeight_in->GetResult()))[19] = -0.01955;
        (*(pWeight_in->GetResult()))[20] = -0.02089;    (*(pWeight_in->GetResult()))[21] = 0.1231;    (*(pWeight_in->GetResult()))[22] = -0.06875;   (*(pWeight_in->GetResult()))[23] = -0.006186;   (*(pWeight_in->GetResult()))[24] = 0.01286;
        (*(pWeight_in->GetResult()))[25] = -0.05449;    (*(pWeight_in->GetResult()))[26] = 0.05843;   (*(pWeight_in->GetResult()))[27] = 0.01405;   (*(pWeight_in->GetResult()))[28] = 0.04691; (*(pWeight_in->GetResult()))[29] = -0.1469;
        (*(pWeight_in->GetResult()))[30] = 0.006045;   (*(pWeight_in->GetResult()))[31] = 0.03382;  (*(pWeight_in->GetResult()))[32] = -0.03095;  (*(pWeight_in->GetResult()))[33] = -0.00622;  (*(pWeight_in->GetResult()))[34] = -0.05395;
        (*(pWeight_in->GetResult()))[35] = -0.07982;   (*(pWeight_in->GetResult()))[36] = -0.1226;  (*(pWeight_in->GetResult()))[37] = 0.003088;  (*(pWeight_in->GetResult()))[38] = 0.05011;  (*(pWeight_in->GetResult()))[39] = -0.0666;
        (*(pWeight_in->GetResult()))[40] = 0.1533;    (*(pWeight_in->GetResult()))[41] = 0.1992;   (*(pWeight_in->GetResult()))[42] = 0.118;   (*(pWeight_in->GetResult()))[43] = -0.08967; (*(pWeight_in->GetResult()))[44] = -0.04522;
        (*(pWeight_in->GetResult()))[45] = -0.1241;    (*(pWeight_in->GetResult()))[46] = 0.05166;   (*(pWeight_in->GetResult()))[47] = -0.03945;   (*(pWeight_in->GetResult()))[48] = -0.1026; (*(pWeight_in->GetResult()))[49] = -0.106;


        (*(pWeight_out->GetResult()))[0] = -0.171;    (*(pWeight_out->GetResult()))[1] = 0.03814;    (*(pWeight_out->GetResult()))[2] = 0.09255;   (*(pWeight_out->GetResult()))[3] = -0.2048;   (*(pWeight_out->GetResult()))[4] = -0.05483;
        (*(pWeight_out->GetResult()))[5] = -0.2389;    (*(pWeight_out->GetResult()))[6] = 0.01982;    (*(pWeight_out->GetResult()))[7] = -0.02456;   (*(pWeight_out->GetResult()))[8] = 0.1349;   (*(pWeight_out->GetResult()))[9] = 0.1668;
        (*(pWeight_out->GetResult()))[10] = -0.05184;   (*(pWeight_out->GetResult()))[11] =-0.1789;   (*(pWeight_out->GetResult()))[12] = -0.1948;  (*(pWeight_out->GetResult()))[13] = -0.05536;  (*(pWeight_out->GetResult()))[14] = -0.02107;
        (*(pWeight_out->GetResult()))[15] = 0.1096;   (*(pWeight_out->GetResult()))[16] = -0.1855;   (*(pWeight_out->GetResult()))[17] = 0.06059;  (*(pWeight_out->GetResult()))[18] = -0.01672;  (*(pWeight_out->GetResult()))[19] = -0.2082;
        (*(pWeight_out->GetResult()))[20] = -0.001693;    (*(pWeight_out->GetResult()))[21] = 0.06907;    (*(pWeight_out->GetResult()))[22] = 0.1438;   (*(pWeight_out->GetResult()))[23] = 0.009881;   (*(pWeight_out->GetResult()))[24] = 0.1443;
        (*(pWeight_out->GetResult()))[25] = 0.02535;    (*(pWeight_out->GetResult()))[26] = -0.07795;   (*(pWeight_out->GetResult()))[27] = -0.1505;   (*(pWeight_out->GetResult()))[28] = -0.06751; (*(pWeight_out->GetResult()))[29] = -0.08182;
        (*(pWeight_out->GetResult()))[30] = 0.138;   (*(pWeight_out->GetResult()))[31] = 0.09633;  (*(pWeight_out->GetResult()))[32] = 0.1917;  (*(pWeight_out->GetResult()))[33] = 0.01659;  (*(pWeight_out->GetResult()))[34] = 0.01897;
        (*(pWeight_out->GetResult()))[35] = 0.00445;   (*(pWeight_out->GetResult()))[36] = 0.0572;  (*(pWeight_out->GetResult()))[37] = 0.2901;  (*(pWeight_out->GetResult()))[38] = 0.1272;  (*(pWeight_out->GetResult()))[39] = 0.1541;
        (*(pWeight_out->GetResult()))[40] = 0.04025;    (*(pWeight_out->GetResult()))[41] = 0.0128;   (*(pWeight_out->GetResult()))[42] = -0.03881;   (*(pWeight_out->GetResult()))[43] = 0.033; (*(pWeight_out->GetResult()))[44] = -0.1318;
        (*(pWeight_out->GetResult()))[45] = 0.04346;    (*(pWeight_out->GetResult()))[46] = 0.08447;   (*(pWeight_out->GetResult()))[47] = -0.05661;   (*(pWeight_out->GetResult()))[48] = -0.1369; (*(pWeight_out->GetResult()))[49] = -0.001039;
*/


        //debugging을 위해서 weight값 설정하기!!!


        out = new SkipGram<DTYPE>(out, pWeight_in, pWeight_out, "SKIPGRAM_Layer");

        this->AnalyzeGraph(out);

        return TRUE;
    }

};


#endif  // __SKIPGRAM_LAYER__
