#ifndef __ENCODER__
#define __ENCODER__    value

#include "../Module.hpp"


template<typename DTYPE> class Encoder : public Module<DTYPE>{
private:

    int timesize;

public:

    Encoder(Operator<DTYPE> *pInput, int vocablength, int embeddingDim, int hiddensize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, vocablength, embeddingDim, hiddensize, use_bias, pName);
    }


    virtual ~Encoder() {}

    int Alloc(Operator<DTYPE> *pInput, int vocablength, int embeddingDim, int hiddensize, int use_bias, std::string pName) {

        timesize = pInput->GetResult()->GetTimeSize();
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;


        //embedding 추가???
        //out = new Embedding<DTYPE>(pWeight_in, out, "embedding");
        out = new EmbeddingLayer<float>(out, vocablength, embeddingDim, "Embedding");


        //------------------------------RNN-------------------------
        // Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, embeddingDim, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        // Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);
        // Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);
        // out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias, "Encoder RNN");

        out = new RecurrentLayer<float>(out, embeddingDim, hiddensize, 10, NULL, use_bias, "Recur_1");

        //------------------------------LSTM-------------------------
        // Tensorholder<DTYPE> *pWeight_IG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, embeddingDim, 0.0, 0.01), "LSTMLayer_pWeight_IG_" + pName);
        // Tensorholder<DTYPE> *pWeight_HG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HG_" + pName);
        // Tensorholder<DTYPE> *lstmBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 4*hiddensize, 0.f), "RNN_Bias_f" + pName);
        // out = new SeqLSTM2<DTYPE>(out, pWeight_IG, pWeight_HG, lstmBias);


        //------------------------------GRU-------------------------
        //weight
        // Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, embeddingDim, 0.0, 0.01), "GRULayer_pWeight_IG_" + pName);
        // Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        // Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, embeddingDim, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        // Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        // //bias
        // Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "RNN_Bias_f" + pName);
        // Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_f" + pName);
        // out = new SeqGRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias);

        //out = new GRULayer<float>(out, embeddingDim, hiddensize, outputsize, m_initHiddenTensorholder, TRUE, "Recur_1");

        this->AnalyzeGraph(out);

        return TRUE;
    }

/*
    //기존 BPTT 방식의 forward, backward

    //m_numOfExcutableOperator 이게 private로 되어있어서!!! 그래서 접근이 불가능!!!
    int ForwardPropagate(int pTime=0) {

        // std::cout<<'\n';
         // std::cout<<"encoder Forward 호출"<<'\n';
        // std::cout<<timesize<<'\n';
        //std::cout<<this->GetName()<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=0; ti<timesize; ti++){
            for (int i = 0; i < numOfExcutableOperator; i++) {
                (*ExcutableOperator)[i]->ForwardPropagate(ti);
            }
        }

        // std::cout<<"Encoder Forwrad 결과"<<'\n';
        // std::cout<<this->GetResult()->GetShape()<<'\n';
        // std::cout<<this->GetResult()<<'\n';

        return TRUE;
    }

    int BackPropagate(int pTime=0) {

        //std::cout<<"****************encoder Backward 호출****************"<<'\n';

        // std::cout<<"초기 gradient 값"<<'\n';
        // std::cout<<this->GetGradient()->GetShape()<<'\n';
        // std::cout<<this->GetGradient()<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=timesize-1; ti>=0; ti--){
            for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
                (*ExcutableOperator)[i]->BackPropagate(ti);
            }
        }

        //std::cout<<"encoder backward 호출 완료"<<'\n';

        // std::cout<<"Encoder Backward 호출 완료 후 gradient 값"<<'\n';
        // //std::cout<<this->GetResult()
        // std::cout<<this->GetGradient()->GetShape()<<'\n';
        // std::cout<<this->GetGradient()<<'\n';

        return TRUE;
    }
*/

    //새로 만든 seq2seqBPTT를 위한 forwrad, backward
    //이거 없애도 되는거 같은데!!!....???                                        !!! 중요!!! 여기는 없어도 되는거 같음!!!
    int ForwardPropagate(int pTime=0) {


        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++) {
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);
        }

        return TRUE;
    }

    int BackPropagate(int pTime=0) {

        //std::cout<<"****************encoder Backward 호출****************"<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
        }



        return TRUE;
    }




};



#endif  // __ENCODER__
