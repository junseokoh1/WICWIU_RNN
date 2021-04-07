#ifndef EMBEDDINGTEST_H_
#define EMBEDDINGTEST_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class EmbeddingTest : public Operator<DTYPE>{
private:

     Operator<DTYPE> *m_aWeight;
     Operator<DTYPE> *m_aNewWord;

public:

    EmbeddingTest(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "EmbeddingTest::EmbeddingTest(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeight);
    }

    virtual ~EmbeddingTest() {
        #ifdef __DEBUG__
        std::cout << "EmbeddingTest::~EmbeddingTest()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight) {

        float len = 0.0;

        int timesize           = pInput->GetResult()->GetTimeSize();
        int batchsize          = pInput->GetResult()->GetBatchSize();
        int vocabsize          = pWeight->GetResult()->GetRowSize();
        int embeddingDim       = pWeight->GetResult()->GetColSize();

        //weight 값 수정하기 위한 작업
        Tensor<DTYPE> *weightTensor   = pWeight->GetResult();
        Shape *weightShape               = weightTensor->GetShape();

        //weight에는 batch가 없지!!!
        for(int vo = 0; vo < vocabsize; vo++){

            len = 0;
            for(int em = 0; em < embeddingDim; em++){
                len += (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, em)] * (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, em)];
            }

            len = sqrt(len);
            for(int em = 0; em < embeddingDim; em++){
                (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, em)] /= len;
            }
        }

        //변경된 값 갖고 이제 설정해주기!
        m_aWeight            = new Tensorholder<DTYPE>(weightTensor, "result");
        m_aNewWord           = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, embeddingDim), "result");

        //해당 operator의 result 설정해주기
        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));


        return TRUE;
    }

    void Delete() {

    }

    int ForwardPropagate(int pTime = 0) {

        float dist = 0, bestDist = 0;
        int bestWordIndex = 0, first = 0, second = 0, third = 0;

        Tensor<DTYPE> *inputTensor    = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *wordTensor     = m_aNewWord->GetResult();                //2-1+3의 결과를 저장하기위한 공간!
        Tensor<DTYPE> *weightTensor   = m_aWeight->GetResult();
        Tensor<DTYPE> *resultTensor   = this->GetResult();

        Shape *inputShape    = inputTensor->GetShape();
        Shape *wordShape     = wordTensor->GetShape();
        Shape *weightShape   = weightTensor->GetShape();
        Shape *resultShape   = resultTensor->GetShape();

        int batchsize      = wordTensor->GetBatchSize();
        int embeddingDim   = wordTensor->GetColSize();
        int vocabsize      = weightTensor->GetRowSize();

        // std::cout<<wordTensor<<'\n';

        // std::cout<<"embeddingDim : "<<embeddingDim<<" vocabsize : "<<vocabsize<<'\n';

        for(int ba=0; ba < batchsize; ba++){

              //2 - 1 + 3 계산해서 벡터에 저장
              first   = (*inputTensor)[Index5D(inputShape, 0, ba, 0, 0, 0)];
              second  = (*inputTensor)[Index5D(inputShape, 0, ba, 0, 0, 1)];
              third   = (*inputTensor)[Index5D(inputShape, 0, ba, 0, 0, 2)];

              for(int em = 0; em < embeddingDim; em++){
                  (*wordTensor)[Index5D(wordShape, 0, ba, 0, 0, em)]
                        = (*weightTensor)[Index5D(weightShape, 0, 0, 0, second, em)]
                           - (*weightTensor)[Index5D(weightShape, 0, 0, 0, first, em)]
                           + (*weightTensor)[Index5D(weightShape, 0, 0, 0, third, em)];

                 // std::cout<<(*weightTensor)[Index5D(weightShape, 0, 0, 0, second, em)]
                 //    - (*weightTensor)[Index5D(weightShape, 0, 0, 0, first, em)]<<" ";
              }

              bestDist = 0;

              for(int vo=0; vo<vocabsize; vo++){

                    dist = 0;

                    for(int em=0; em<embeddingDim; em++)
                        dist +=  (*wordTensor)[Index5D(wordShape, 0, ba, 0, 0, em)] * (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, em)];

                    if(dist > bestDist){
                        bestDist = dist;
                        bestWordIndex = vo;
                    }
              }
              //결과값 저장해주기!!!
              (*resultTensor)[Index5D(resultShape, 0, ba, 0, 0, 0)] = bestWordIndex;
        }

        return TRUE;
    }


    int ResetResult() {

        m_aNewWord->ResetResult();
        //한번 설정해준 weight값은 변경하면 안되기 때문에!!! reset을 안해준다!!!

        Tensor<DTYPE> *result  = this->GetResult();
        result->Reset();
    }



    // //방법2
    // int Test(int pTime = 0) {
    //
    //     float dist = 0;
    //
    //     Operator<DTYPE> *pWeight_in  = this->GetInput()[1];
    //
    //     int vocabNum = pWeight_in->GetResult()->GetRowSize();
    //     int embeddingDim = pWeight_in->GetResult()->GetColSize();
    //
    //     //한번 만 호출!!! 초기 값 설정을 위해 필요!!!
    //     if(flag == 0){
    //           float len = 0;
    //
    //           m_aTestWeight = new Tensorholder<DTYPE>(pWeight_in->GetTensor(), "testWeight");
    //
    //           Tensor<DTYPE> *weightTensor    = m_aTestWeight->GetResult();
    //           Shape *weightShape             = weightTensor->GetShape();
    //
    //           for(int vo=0; vo<vocabNum; vo++){
    //               for(int embed=0; embed<embeddingDim; embed++){
    //                   len += (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, embed)]*(*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, embed)];
    //               }
    //
    //               len = sqrt(len);
    //
    //               for(int embed=0; embed<embeddingDim; embed++){
    //                   (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, embed)] /= len;
    //               }
    //           }
    //           flag = 1;
    //     }
    //
    //     m_aTestIndex->ForwardPropagate(pTime);
    //
    //     for(int vo=0; vo<vocabNum; vo++){
    //
    //         //입력이랑 똑같은 index는 pass!!!
    //
    //         dist = 0;
    //
    //         for(int embed=0; embed<embeddingDim; embed++)
    //             dist +=  ??? * (*weightTensor)[Index5D(weightShape, 0, 0, 0, vo, embed)];
    //
    //     }
    //
    //     return TRUE;
    // }


};


#endif  // EmbeddingTest_H_
