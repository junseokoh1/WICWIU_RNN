#ifndef __ATTENTIONWEIGHT_HPP__
#define __ATTENTIONWEIGHT_HPP__


#include "../Module.hpp"


template<typename DTYPE> class DotAttentionWeight : public Module<DTYPE> {
private:
public:
  DotAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName = "NO NAME");
  int       Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName);
  virtual int SetQuery(Operator<DTYPE> *pQuery);
};


template<typename DTYPE> DotAttentionWeight<DTYPE>::DotAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) : Module<DTYPE>(pName) {
  #ifdef __DEBUG__
  std::cout << "DotAttentionWeight<DTYPE>::DotAttentionWeight(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, bool , std::string )" << '\n';
  #endif  // __DEBUG__

  Alloc(pKey, pQuery, pMask, pName);
}

template<typename DTYPE> int DotAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "DotAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *, bool , std::string )" << '\n';
    #endif  // __DEBUG__

    this->SetInput(3, pKey, pQuery, pMask);     //이거 확인해보기!!!

    Operator<DTYPE> *out = NULL;

    out = new DotSimilarity<DTYPE>(pKey, pQuery, pName+"_similarity");

    if(pMask) {
      out = new MaskedFill<DTYPE>(out, pMask, pName+"_pMask");
    }

    out = new Softmax<DTYPE>(out, pName+"_attention_weight");

    this->AnalyzeGraph(out);

    return TRUE;
}

template<typename DTYPE> int DotAttentionWeight<DTYPE>::SetQuery(Operator<DTYPE> * pQuery){

    std::cout<<"DotAttentionWeight SetQuery"<<'\n';

    // int numOfExcutableOperator = this->GetNumOfExcutableOperator();
    Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();
    (*ExcutableOperator)[0]->SetQuery(pQuery);


}



//key   : encoder hidden
//query : decoder hidden
template<typename DTYPE> class BahdanauAttentionWeight : public Module<DTYPE> {
private:

public:

    BahdanauAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) : Module<DTYPE>(pName) {
        Alloc(pKey, pQuery, pMask, pName);
    }

    virtual ~BahdanauAttentionWeight() {}

    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) {
        this->SetInput(3, pKey, pQuery, pMask);     //이거 확인해보기!!!


        //weight shape을 위한 값 얻기!
        //weight은 ti, ba는 고려하지 않고 다 동일한 값을 사용하지!
        int channelSize    = pKey->GetResult()->GetChannelSize();
        int rowSize        = pKey->GetResult()->GetRowSize();
        int EncoderColSize = pKey->GetResult()->GetColSize();
        int DecoderColSize = pQuery->GetResult()->GetColSize();

        //weight 생성
        Tensorholder<DTYPE> *pWeightV = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, 1, DecoderColSize, 0.0, 0.01), "Bahdanau_Weight_V_" + pName);
        Tensorholder<DTYPE> *pWeightW = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, DecoderColSize, DecoderColSize, 0.0, 0.01), "Bahdanau_Weight_W_" + pName);
        Tensorholder<DTYPE> *pWeightU = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, channelSize, DecoderColSize, EncoderColSize, 0.0, 0.01), "Bahdanau_Weight_U_" + pName);

        // Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);

        Operator<DTYPE> *out = NULL;

        out = new ConcatSimilarity<DTYPE>(pKey, pWeightV, pWeightW, pWeightU, pQuery, pName+"_similarity");

        // std::cout<<'\n'<<"----ConcatSimilarity 의 inputContainer"<<'\n';
        // Container<Operator<DTYPE> *> *attention_C = out->GetInputContainer();
        // std::cout<<attention_C->GetSize()<<'\n';
        // std::cout<<(*attention_C)[0]->GetName()<<'\n';
        // std::cout<<(*attention_C)[1]->GetName()<<'\n';
        // std::cout<<(*attention_C)[2]->GetName()<<'\n';
        // std::cout<<(*attention_C)[3]->GetName()<<'\n';
        // std::cout<<(*attention_C)[4]->GetName()<<'\n';

        // MatMul    //encoder timesize만큼 다 돌아야됨...
        //
        // newOp
        //
        // Add       //add도 기존의 op 사용 불가능.... 아니면 모든 time에 똑같은 값을 복사해서 time만큼 돌려보리면됨....
        //
        // Tanh      //이것도 time만큼 다 돌아야되네....ㅋㅋㅋㅋㅋㅋ
        //
        // matmul   //이것도 encoder timesize만큼 다 돌아야됨...
        // //그리고 time wise로 존재하는 값을 col wise로 합쳐야됨....

        if(pMask) {
          out = new MaskedFill<DTYPE>(out, pMask, pName+"_pMask");
        }

        out = new Softmax<DTYPE>(out, pName+"_attention_weight");

        this->AnalyzeGraph(out);



        return TRUE;
    }

    int SetQuery(Operator<DTYPE> * pQuery){

        std::cout<<"BahdanauAttentionWeight SetQuery"<<'\n';

        // int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();
        (*ExcutableOperator)[0]->SetQuery(pQuery);

    }
};


#endif
