#ifndef DOTSIMILARITY_H_
#define DOTSIMILARITY_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class DotSimilarity : public Operator<DTYPE>{

public:

    DotSimilarity(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pKey, pQuery, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "DotSimilarity::DotSimilarity(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pKey, pQuery);
    }


    virtual ~DotSimilarity() {
        #ifdef __DEBUG__
        std::cout << "DotSimilarity::~DotSimilarity()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    //pQuery - t 시점의 decoder hidden, recurent의 result를 가져올거임!
    //pKey - 모든 Encoder의 hidden 값!
    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery) {
        #ifdef __DEBUG__
        std::cout << "DotSimilarity::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pQuery->GetResult()->GetTimeSize();
        int batchsize   = pQuery->GetResult()->GetBatchSize();
        int channelsize = pQuery->GetResult()->GetChannelSize();
        int rowsize     = pQuery->GetResult()->GetRowSize();
        int colsize     = pKey->GetResult()->GetTimeSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }


    void Delete() {
    }

    virtual int SetQuery(Operator<DTYPE>* pQuery){
        std::cout<<"DotSimilarity SetQuery"<<'\n';
        this->GetInputContainer()->Pop(this->GetInputContainer()->GetLast());
        this->GetInputContainer()->Push(pQuery);
    }

    //pQuery - t 시점의 decoder hidden, recurent의 result를 가져올거임!
    //pKey - 모든 Encoder의 hidden 값!
    //forward는 확인 함!!! 잘 작동! -excel로 확인!
    //batch에 대해서는 확인안해봄!....
    int ForwardPropagate(int pTime = 0) {

        // std::cout<<"DotSimilarity Forward "<<pTime<<'\n';

        //연결 확인해보기
        // Container<Operator<DTYPE> *> *inputcontainer = this->GetInputContainer();
        // std::cout<<(*inputcontainer)[0]->GetName()<<'\n';
        // std::cout<<(*inputcontainer)[1]->GetName()<<'\n';

        Tensor<DTYPE> *key = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *query  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *keyShape = key->GetShape();
        Shape *queryShape  = query->GetShape();
        Shape *resultShape = result->GetShape();

        // std::cout<<this->GetInput()[1]->GetName()<<'\n';

        // std::cout<<"------key(encoder hidden) shape"<<'\n';
        // std::cout<<keyShape<<'\n';
        // std::cout<<"------Query(decoder hidden) shape"<<'\n';
        // std::cout<<queryShape<<'\n';
        // std::cout<<"------result shape"<<'\n';
        // std::cout<<resultShape<<'\n';

        int keytimesize = key->GetTimeSize();
        int batchsize   = key->GetBatchSize();
        int colsize     = key->GetColSize();

        for(int ti=0; ti< keytimesize; ti++){
            // std::cout<<ti<<" ";
            for(int ba = 0; ba < batchsize; ba++) {
                // std::cout<<'\n'<<ba<<" ";
                for(int co = 0; co < colsize; co++) {
                    // std::cout<<ti<<" "<<ba<<" "<<co<<'\n';
                    (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)] +=
                        (*query)[Index5D(queryShape, pTime, ba, 0, 0, co)] * (*key)[Index5D(keyShape, ti, ba, 0, 0, co)];
                }
            }
        }

        // std::cout<<'\n';

        return TRUE;
    }

    // excel로는 확인 안했는데 잘 동작하는거 같음!!!
    //pQuery - t 시점의 decoder hidden, recurent의 result를 가져올거임!
    //pKey - 모든 Encoder의 hidden 값!
    int BackPropagate(int pTime = 0) {

      //std::cout<<"---------------------DotSimilarity Forward----------------------"<<pTime<<'\n';

      Tensor<DTYPE> *key = this->GetInput()[0]->GetGradient();
      Tensor<DTYPE> *query  = this->GetInput()[1]->GetGradient();
      Tensor<DTYPE> *result = this->GetGradient();

      Shape *keyShape = key->GetShape();
      Shape *queryShape  = query->GetShape();
      Shape *resultShape = result->GetShape();


      int keytimesize = key->GetTimeSize();
      int batchsize   = key->GetBatchSize();
      int colsize     = key->GetColSize();


      for(int ti=0; ti< keytimesize; ti++){
          for(int ba = 0; ba < batchsize; ba++) {
              for(int co = 0; co < colsize; co++) {

                  (*query)[Index5D(queryShape, pTime, ba, 0, 0, co)] +=
                      (*key)[Index5D(keyShape, ti, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)];

                  (*key)[Index5D(keyShape, ti, ba, 0, 0, co)] +=
                      (*query)[Index5D(queryShape, pTime, ba, 0, 0, co)] * (*result)[Index5D(resultShape, pTime, ba, 0, 0, ti)];

              }
          }
      }

      //std::cout<<query<<'\n';

        return TRUE;
    }

};


#endif  // DOTSIMILARITY_H_
