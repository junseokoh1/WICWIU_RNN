#ifndef ATTENTIONBYMODUEL_H_
#define ATTENTIONBYMODUEL_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class AttentionByModule : public Operator<DTYPE>{
private:

public:
    AttentionByModule(Operator<DTYPE> *pAttentionWeight, Operator<DTYPE> *pValue, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pAttentionWeight, pValue, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "AttentionByModule::AttentionByModule(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pAttentionWeight, pValue);
    }


    virtual ~AttentionByModule() {
        #ifdef __DEBUG__
        std::cout << "AttentionByModule::~AttentionByModule()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    //value - 인코더
    //목표는 context vector를 구하는거!!!
    //result의 shape은 value를 따라가는게 맞음!!!
    int Alloc(Operator<DTYPE> *pAttentionWeight, Operator<DTYPE> *pValue) {
        #ifdef __DEBUG__
        std::cout << "AttentionByModule::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize      = pAttentionWeight->GetResult()->GetTimeSize();
        int batchsize     = pAttentionWeight->GetResult()->GetBatchSize();
        int channelsize   = pAttentionWeight->GetResult()->GetChannelSize();
        int rowsize       = pAttentionWeight->GetResult()->GetRowSize();
        int colsize       = pValue->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    void Delete() { }


    int ForwardPropagate(int pTime = 0) {

        //std::cout<<"AttentionByModule Forward 호출 "<<pTime<<'\n';

        Tensor<DTYPE> *attenionWeight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *value  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        // std::cout<<attenionWeight<<'\n';

        //context vector = result, value를 사용해서 this->result에 저장!
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int valueTimeSize = value->GetTimeSize();

        Shape *weightShape = attenionWeight->GetShape();
        Shape *valueShape = value->GetShape();
        Shape *resultShape = result->GetShape();

        //j랑 ba랑 위치 바꿔야되나?...
        //value = h
        for(int ti=0; ti< valueTimeSize; ti++){
            for(int ba = 0; ba < batchsize; ba++) {
                for(int co = 0; co < colsize; co++) {
                    (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)] +=
                        (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] * (*value)[Index5D(valueShape, ti, ba, 0, 0, co)];
                }
            }
        }

        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        //std::cout<<"AttentionByModule BackPropagate 호출 "<<pTime<<'\n';

        Tensor<DTYPE> *attenionWeight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *value  = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *result = this->GetGradient();

        //context vector = result, value를 사용해서 this->result에 저장!
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int valueTimeSize = value->GetTimeSize();

        Shape *weightShape = attenionWeight->GetShape();
        Shape *valueShape = value->GetShape();
        Shape *resultShape = result->GetShape();

        //value = h
        for(int ti=0; ti< valueTimeSize; ti++){
            for(int ba = 0; ba < batchsize; ba++) {
                for(int co = 0; co < colsize; co++) {
                    (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] +=
                        (*value)[Index5D(valueShape, ti, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)];

                    (*value)[Index5D(valueShape, ti, ba, 0, 0, co)] +=
                        (*attenionWeight)[Index5D(weightShape, pTime, ba, 0, 0, ti)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, co)];
                }
            }
        }


        return TRUE;
    }

};


#endif  // ATTENTIONBYMODUEL_H_
