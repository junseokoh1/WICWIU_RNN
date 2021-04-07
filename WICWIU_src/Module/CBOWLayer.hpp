#ifndef __CBOW_LAYER__
#define __CBOW_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class CBOWLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief CBOWLayer 클래스 생성자
    @details CBOWLayer 클래스의 Alloc 함수를 호출한다.*/
    CBOWLayer(Operator<DTYPE> *pInput, int vocabsize, int hiddensize, int windowsize, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, vocabsize, hiddensize, windowsize, pName);
    }

    /*!
    @brief CBOWLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~CBOWLayer() {}


    int Alloc(Operator<DTYPE> *pInput, int vocabsize, int hiddensize, int windowsize, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //------------------------------weight 생성-------------------------
        //Win 여기서 window 사이즈만큼 곱하기 안해주는 이유 : input에서 잘라서 값 복사해서 처리해주기???
        Tensorholder<DTYPE> *pWeight_in = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, vocabsize, 0.0, 0.01), "CBOWLayer_pWeight_in_" + pName);

        //이거는... 복사가 아니라 입력을 하나로 합쳐서 처리.... 그러면... 동일한 tensor를 생성하는 새로운 방법이 필요....
        //Tensorholder<DTYPE> *pWeight_in = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, vocabsize*windowsize, 0.0, 0.01), "CBOWLayer_pWeight_in_" + pName);

        Tensorholder<DTYPE> *pWeight_out = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, hiddensize, 0.0, 0.01), "CBOWLayer_pWeight_out_" + pName);


        out = new CBOW<DTYPE>(out, pWeight_in, pWeight_out, "CBOW_Layer");
        //embedding추가한거!

        //Tensorholder<DTYPE> *pWeight_in = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, hiddensize, 0.0, 0.01), "CBOWLayer_pWeight_in_" + pName);        //이거는 embedding을 사용해서 이렇게 넣은거!!!

        //out = new CBOWEmbedding<DTYPE>(out, pWeight_in, pWeight_out, "CBOW_Layer");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __CBOW_LAYER__
