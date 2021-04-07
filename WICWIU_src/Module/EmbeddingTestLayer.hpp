#ifndef __EMBEDDINGTEST_LAYER__
#define __EMBEDDINGTEST_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class EmbeddingTestLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief CBOWLayer 클래스 생성자
    @details CBOWLayer 클래스의 Alloc 함수를 호출한다.*/
    EmbeddingTestLayer(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pWeight, pName);
    }

    /*!
    @brief CBOWLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~EmbeddingTestLayer() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        out = new EmbeddingTest<DTYPE>(out, pWeight, "EmbeddingTest_Layer_" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __EMBEDDINGTEST_LAYER__
