#ifndef ATTENTION_H_
#define ATTENTION_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class Attention : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_attentionLogit;
    Operator<DTYPE> *m_attentionWeight;

public:
    Attention(Operator<DTYPE> *pKey, Operator<DTYPE> *pValue, int maxTimesize, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pKey, pValue, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Attention::Attention(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pKey, pValue, maxTimesize);
    }


    virtual ~Attention() {
        #ifdef __DEBUG__
        std::cout << "Attention::~Attention()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    //query - 디코더
    //key - 인코더
    //value - 인코더

    //result의 shape은 value를 따라가는게 맞음!!!
    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pValue, int maxTimesize) {
        #ifdef __DEBUG__
        std::cout << "Attention::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int valuetimesize = pValue->GetResult()->GetTimeSize();
        int batchsize     = pValue->GetResult()->GetBatchSize();
        int channelsize   = pValue->GetResult()->GetChannelSize();
        int rowsize       = pValue->GetResult()->GetRowSize();
        int colsize       = pValue->GetResult()->GetColSize();       //이거 아니지....  !!!! 이거 hidden size이여야 되는거 같아!!!

        m_attentionLogit    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(maxTimesize, batchsize, channelsize, rowsize, valuetimesize), "attention_logit");
        m_attentionWeight   = new Softmax<DTYPE>(m_attentionLogit, "attention_weight");

        this->SetResult(new Tensor<DTYPE>(maxTimesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(maxTimesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    void Delete() { }

    //virtual int GetSimilarity(int pTime, Operator<DTYPE> *pAttentionLogit){}

    virtual int ForwardSimilarity(int pTime, Operator<DTYPE> *pAttentionLogit){}
    virtual int BackwardSimilarity(int pTime, Operator<DTYPE> *pAttentionLogit){}

    int ForwardPropagate(int pTime = 0) {

        std::cout<<"attention forward함수 호출"<<'\n';

        //similarity - e
        ForwardSimilarity(pTime, m_attentionLogit);

        std::cout<<"similarity 함수 호출 성공"<<'\n';

        //a
        m_attentionWeight->ForwardPropagate(pTime);

        Tensor<DTYPE> *value  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();
        Tensor<DTYPE> *attenionWeight = m_attentionWeight->GetResult();

        std::cout<<attenionWeight<<'\n';

        //context vector = result, value를 사용해서 this->result에 저장!
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int valuetimesize = value->GetTimeSize();

        Shape *weightShape = attenionWeight->GetShape();
        Shape *valueShape = value->GetShape();
        Shape *resultShape = result->GetShape();

        //j랑 ba랑 위치 바꿔야되나?...
        //value = h
        for(int ti=0; ti< valuetimesize; ti++){
            for(int ba = 0; ba < batchsize; ba++) {
                for(int co = 0; co < colsize; co++) {
                    (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)] +=
                        (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] * (*value)[Index5D(valueShape, ti, ba, 0, 0, co)];
                }
            }
        }

        std::cout<<"Attention forward 종료"<<'\n';
        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *value  = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *result = this->GetGradient();
        Tensor<DTYPE> *attenionWeight = m_attentionWeight->GetGradient();

        //context vector = result, value를 사용해서 this->result에 저장!
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int valuetimesize = value->GetTimeSize();

        Shape *weightShape = attenionWeight->GetShape();
        Shape *valueShape = value->GetShape();
        Shape *resultShape = result->GetShape();

        //value = h
        for(int ti=0; ti< valuetimesize; ti++){
            for(int ba = 0; ba < batchsize; ba++) {
                for(int co = 0; co < colsize; co++) {
                    (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] +=
                        (*value)[Index5D(valueShape, ti, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)];

                    (*value)[Index5D(valueShape, ti, ba, 0, 0, co)] +=
                        (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)];
                }
            }
        }

        m_attentionWeight->BackPropagate(pTime);

        BackwardSimilarity(pTime, m_attentionLogit);


        return TRUE;
    }

};


#endif  // ATTENTION_H_
