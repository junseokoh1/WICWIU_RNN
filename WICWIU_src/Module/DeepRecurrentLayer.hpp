#ifndef __DEEPRECURRENT_LAYER__
#define __DEEPRECURRENT_LAYER__    value            //여기 부분 VLAUE하고 저렇게 하는게 맞는지 확인 할 것!!!!

#include "../Module.hpp"


template<typename DTYPE> class DeepRecurrentLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief DeepRecurrentLayer 클래스 생성자
    @details DeepRecurrentLayer 클래스의 Alloc 함수를 호출한다.*/
    DeepRecurrentLayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, use_bias, pName);
    }

    /*!
    @brief DeepRecurrentLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~DeepRecurrentLayer() {}

    /*!
    @brief DeepRecurrentLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @param pInput
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //--------------------------------------------초기화 방법. 추후 필히 수정!!!!!!!!!!!
        float xavier_i = 1/sqrt(inputsize);
        float xavier_h = 1/sqrt(hiddensize);



        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        //pWeight_x2h2는 결국 Deep RNN에서 보면 hidden에서 hidden으로 가는거기 때문에 사이즈가 hiddensize, hiddensize 이여야지
        Tensorholder<DTYPE> *pWeight_x2h2 = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h1 = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h2 = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);

        //recurrent 내에 bias 추가
        Tensorholder<DTYPE> *rBias1 = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias1_" + pName);
        Tensorholder<DTYPE> *rBias2 = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias2_" + pName);



        out = new Recurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h1, rBias1);
        out = new Recurrent<DTYPE>(out, pWeight_x2h2, pWeight_h2h2, rBias2);

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");



        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __RECURRENT_LAYER__
