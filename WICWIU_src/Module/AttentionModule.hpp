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

        //이것도.... module 이네.............AttentionWeight.hpp
        Operator<DTYPE> *out = new DotAttentionWeight<DTYPE>(pKey, pQuery, pMask, pName+"_DotAttentionWeight");        //여기 내부에 operator 3개 존재

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
