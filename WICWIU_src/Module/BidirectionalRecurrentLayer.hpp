#ifndef __BIRECURRENT_LAYER__
#define __BIRECURRENT_LAYER__    value

#include "../Module.hpp"

//initHidden에 대해서는 구현 안함....
//뭐... initHidden을 넘겨주면 2개로 있으니깐 쪼개서 각각에 넣어주면 되니깐....

template<typename DTYPE> class BidirectionalRecurrentLayer : public Module<DTYPE>{
private:

    Operator<DTYPE> * reversedInput;

public:
    /*!
    @brief RecurrentLayer 클래스 생성자
    @details RecurrentLayer 클래스의 Alloc 함수를 호출한다.*/
    BidirectionalRecurrentLayer(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden = NULL, int useBias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputSize, hiddenSize, initHidden, useBias, pName);
    }

    virtual ~BidirectionalRecurrentLayer() {}


    int Alloc(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE>* initHidden, int useBias, std::string pName) {
        this->SetInput(pInput);
        // this->SetInput(2, pInput, initHidden);

        Operator<DTYPE> *out = pInput;

        reversedInput = new FlipTimeWise<DTYPE>(pInput, "FLIP");       //stack이면... reverse를 하면 안되고... 그냥 그대로 올라가야됨... 근데 애당초  stack이면 concate을 안하지 않나...

        //weight & bias for forward
        Tensorholder<DTYPE> *pWeight_x2h_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, inputSize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_f_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, hiddenSize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_f_" + pName);
        Tensorholder<DTYPE> *rBias_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddenSize, 0.f), "RNN_Bias_f_" + pName);

        //weight & bias for backward
        Tensorholder<DTYPE> *pWeight_x2h_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, inputSize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_b_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, hiddenSize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_b_" + pName);
        Tensorholder<DTYPE> *rBias_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddenSize, 0.f), "RNN_Bias_b_" + pName);

#ifdef __CUDNN__
        pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, hiddenSize+inputSize+1, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //For 1 bias option
#endif  // __CUDNN__

        Operator<DTYPE> * Fout = new SeqRecurrent<DTYPE>(out, pWeight_x2h_f, pWeight_h2h_f, rBias_f, initHidden);

        Operator<DTYPE> * Bout = new SeqRecurrent<DTYPE>(reversedInput, pWeight_x2h_b, pWeight_h2h_b, rBias_b, initHidden);

        Operator<DTYPE> * concate = new ConcatenateColumnWise<DTYPE>(Fout,Bout, "concatenate");

        this->AnalyzeGraph(concate);

        return TRUE;
    }

};


#endif  // __RECURRENT_LAYER__
