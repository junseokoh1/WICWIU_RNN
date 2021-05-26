#ifndef __ATTENTIONWEIGHT_HPP__
#define __ATTENTIONWEIGHT_HPP__


#include "../Module.hpp"

//d_h : head의 개수!
/*
template<typename DTYPE> class TransformerAttentionWeight : public Module<DTYPE> {
private:
public:
  TransformerAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate = 0.0f, std::string pName = "NO NAME");
  int       Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate, std::string pName);
};


template<typename DTYPE> TransformerAttentionWeight<DTYPE>::TransformerAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate, std::string pName) : Module<DTYPE>(pName) {
  #ifdef __DEBUG__
  std::cout << "TransformerAttentionWeight<DTYPE>::TransformerAttentionWeight(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, bool , std::string )" << '\n';
  #endif  // __DEBUG__

  Alloc(pKey, pQuery, pMask, d_h, droprate, pName);
}

template<typename DTYPE> int TransformerAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, int d_h, float droprate, std::string pName) {
  #ifdef __DEBUG__
  std::cout << "TransformerAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *, bool , std::string )" << '\n';
  #endif  // __DEBUG__

  this->SetInput(3, pKey, pQuery, pMask);

  Operator<DTYPE> *out = NULL;



  // #1. MatMul
  out = new MatMulTest<DTYPE>(new Transpose<DTYPE>(pKey, 3, 4, pName+"_QueryTranspose"), pQuery, pName+"_Key_Query_MatMul");
  // #2. Scale
  out = new Scale<DTYPE>(out, 1/sqrt((DTYPE)d_h), pName+"_Scale");
  // #3. pMask
  if(pMask) {
    out = new MaskedFill<DTYPE>(out, pMask, pName+"_pMask");
  }
  // #4. softmax

  out = new SoftmaxTest<DTYPE>(out, 1e-6f, 4, pName+"_Epsilon");
  //out = new Softmax<DTYPE>(out, "_softmax");

  //out = new Dropout<DTYPE>(out, droprate, pName+"_Dropout");

  this->AnalyzeGraph(out);

  return TRUE;
}
*/


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


/*
template<typename DTYPE> class DotAttentionWeight : public Module<DTYPE> {
private:

public:

  DotAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) : Module<DTYPE>(pName) {

      Alloc(pKey, pQuery, pMask, pName);
  }

  virtual ~DotAttentionWeight() {};

  int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName){

        this->SetInput(3, pKey, pQuery, pMask);

        Operator<DTYPE> *out = new DotSimilarity<DTYPE>(pKey, pQuery, "similarity");

        if(pMask) {
          out = new MaskedFill<DTYPE>(out, pMask, pName+"_pMask");
        }

        out = new Softmax<DTYPE>(out, "attention_weight");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};
*/






#endif
