#ifndef __ATTENTIONMODULE__
#define __ATTENTIONMODULE__    value

#include "../Module.hpp"


template<typename DTYPE> class AttentionModule : public Module<DTYPE>{
private:


public:

    AttentionModule(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pKey, pQuery, pValue, pMask, pName);
    }

    virtual ~AttentionModule() {}

    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask,  std::string pName) {

        this->SetInput(4, pKey, pQuery, pValue, pMask);

        //Luong Dot attention
        // Operator<DTYPE> *out = new DotAttentionWeight<DTYPE>(pKey, pQuery, pMask, pName+"_DotAttentionWeight");        //이것도.... module 이네.............AttentionWeight.hpp   //여기 내부에 operator 3개 존재

        //Bahdanau attetnion
        Operator<DTYPE> *out = new BahdanauAttentionWeight<DTYPE>(pKey, pQuery, pMask, pName+"_DotAttentionWeight");    //이것도 module!       //이거는 attention weight...

        // std::cout<<'\n'<<"-----BahdanauAttentionWeight 의 inputContainer"<<'\n';
        // Container<Operator<DTYPE> *> *attention_C = out->GetInputContainer();
        // std::cout<<attention_C->GetSize()<<'\n';
        // std::cout<<(*attention_C)[0]->GetName()<<'\n';
        // std::cout<<(*attention_C)[1]->GetName()<<'\n';
        // std::cout<<(*attention_C)[2]->GetName()<<'\n';
        // std::cout<<(*attention_C)[3]->GetName()<<'\n';
        // std::cout<<(*attention_C)[4]->GetName()<<'\n';
        // std::cout<<(*attention_C)[5]->GetName()<<'\n';


        out = new AttentionByModule<DTYPE>(out, pValue, pName+"_AttentionByModule");       //value하고 weight값 사용해서 최종 결과인 context vector 구하기

        this->AnalyzeGraph(out);

        return TRUE;
    }

    virtual int SetQuery(Operator<DTYPE> * pQuery){

        std::cout<<"AttentionModule SetQuery"<<'\n';

        // int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();


        (*ExcutableOperator)[0]->SetQuery(pQuery);


    }

};



#endif  // __ATTENTIONMODULE__
