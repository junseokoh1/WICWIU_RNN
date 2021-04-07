#ifndef DOTATTENTION_H_
#define DOTATTENTION_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class DotAttention : public Attention<DTYPE>{
private:
    Operator<DTYPE> *m_Query;

public:
    DotAttention(Operator<DTYPE> *pKey, Operator<DTYPE> *pValue, int maxTimesize, std::string pName, int pLoadflag = TRUE) : Attention<DTYPE>(pKey, pValue, maxTimesize, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "DotAttention::DotAttention(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc();
    }


    virtual ~DotAttention() {
        #ifdef __DEBUG__
        std::cout << "DotAttention::~DotAttention()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc() {
        #ifdef __DEBUG__
        std::cout << "Attention::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        return TRUE;
    }

    void Delete() { }

    //그래프에서도 연결을 해줘야 하는가!!!
    //그래프 상에서 연결해서... gradient가 흐는거는 일단 나중에 생각하자!!!
    int SetQuery(Operator<DTYPE> *pQuery){
        m_Query = pQuery;

        return TRUE;
    }

    int ForwardSimilarity(int pTime, Operator<DTYPE> *pAttentionLogit){

        std::cout<<"similarity 함수 호출"<<'\n';

        Tensor<DTYPE> *query  = m_Query->GetResult();                 //디코더의 hidden  , 이전 time의 hidden값만 사용해야됨!
        Tensor<DTYPE> *key    = this->GetInput()[0]->GetResult();     //인코더의 모든 time의 hidden
        Tensor<DTYPE> *result = pAttentionLogit->GetResult();

        Shape *queryShape  = query->GetShape();
        Shape *keyShape    = key->GetShape();
        Shape *resultShape = result->GetShape();

        int keytimesize = key->GetTimeSize();
        int batchsize   = key->GetBatchSize();
        int colsize     = key->GetColSize();

        std::cout<<"result shape : "<<resultShape<<'\n';
        std::cout<<"query shape : "<<queryShape<<'\n';
        std::cout<<"key shape : "<<keyShape<<'\n';

        //pTime이 0일때 처리하는 방법이 필요!                           -> 이게 왜 필요했지?...
        for(int ti=0; ti< keytimesize; ti++){                                                     //여기서 timesize만큼 돌려주네!!! 중요!!!
            std::cout<<ti<<'\n';
            for(int ba = 0; ba < batchsize; ba++) {
                for(int co = 0; co < colsize; co++) {
                    //pTime = 0일때 처리하기
                    std::cout<<"ba : "<<ba<<" co : "<<co<<'\n';
                    //std::cout<<(*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)]<<'\n';
                    //std::cout<<(*key)[Index5D(keyShape, ti, ba, 0, 0, co)]<<'\n';
                    //std::cout<<(*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)]<<'\n';

                    if(pTime == 0){
                      (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)] += 0;
                    }else{
                      (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)] +=
                          (*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)] * (*key)[Index5D(keyShape, ti, ba, 0, 0, co)];           // pTime-1로 되어있고 그때는 Bahdanau 논문 보고있었어서.... 이 함수는 Bahdanau를 위한 attention함수라고 생각됨!
                    }
                }
            }
        }

    }

    int BackwardSimilarity(int pTime, Operator<DTYPE> *pAttentionLogit){

      Tensor<DTYPE> *query  = m_Query->GetGradient();                 //디코더의 hidden  , 이전 time의 hidden값만 사용해야됨!
      Tensor<DTYPE> *key    = this->GetInput()[0]->GetGradient();     //인코더의 모든 time의 hidden
      Tensor<DTYPE> *result = pAttentionLogit->GetGradient();

      Shape *queryShape  = query->GetShape();
      Shape *keyShape    = key->GetShape();
      Shape *resultShape = result->GetShape();

      int keytimesize = key->GetTimeSize();
      int batchsize   = key->GetBatchSize();
      int colsize     = key->GetColSize();


      //pTime이 0일때 처리하는 방법이 필요!
      for(int ti=0; ti< keytimesize; ti++){
          for(int ba = 0; ba < batchsize; ba++) {
              for(int co = 0; co < colsize; co++) {
                  //pTime = 0일때 처리하기

                  if(pTime !=0){
                    (*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)] +=
                        (*key)[Index5D(keyShape, ti, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)];

                    (*key)[Index5D(keyShape, ti, ba, 0, 0, co)] +=
                        (*query)[Index5D(queryShape, pTime-1, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)];
                  }
              }
          }
      }

    }

};


#endif  // DOTATTENTION_H_
