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

        Operator<DTYPE> *out = new DotAttentionWeight<DTYPE>(pKey, pQuery, pMask);

        out = new AttentionByModule<DTYPE>(out, pValue, pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }

};



#endif  // __ATTENTIONMODULE__
