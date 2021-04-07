#ifndef PADDINGATTENTIONMASKRNN_HPP_
#define PADDINGATTENTIONMASKRNN_HPP_

#include "../Operator.hpp"


/*

  입력으로 index를 준다고 생각을 하고 만든거!!!
  그래서 입력을 one-hot으로 주면 이상하게 됨!

*/

template<typename DTYPE> class PaddingAttentionMaskRNN : public Operator<DTYPE> {
private:
  DTYPE m_paddingTok;
public:
  PaddingAttentionMaskRNN(Operator<DTYPE> *pInput, DTYPE mask = 0.F, std::string Name = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, Name, pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "PaddingAttentionMaskRNN<DTYPE>::PaddingAttentionMaskRNN(Operator<DTYPE> , int , int , int , std::string , int )" << '\n';
    #endif  // __DEBUG__

    m_paddingTok = 0;

    Alloc(pInput, mask);
  }

  int Alloc(Operator<DTYPE> *pInput, int mask) {
    #ifdef __DEBUG__
    std::cout << "PaddingAttentionMaskRNN<DTYPE>::Alloc(Operator<DTYPE> *, int " << '\n';
    #endif  // __DEBUG__

    m_paddingTok = mask;

    int timesize    = pInput->GetResult()->GetTimeSize();
    int batchsize   = pInput->GetResult()->GetBatchSize();
    int channelsize = pInput->GetResult()->GetChannelSize();
    int rowsize     = pInput->GetResult()->GetRowSize();
    //int colsize     = pInput->GetResult()->GetTimeSize();


    //channel하고 row는 어차피 1임!
    //모든 time에 대해서 동일한 mask를 사용하기 때문에!
    this->SetResult(new Tensor<DTYPE>(1, batchsize, channelsize, rowsize, timesize));
    this->SetDelta(new Tensor<DTYPE>(1, batchsize, channelsize, rowsize, timesize));

  }


  // 0 : 실제 값이 있다
  // 1 : padding이 되어 있다
  int ForwardPropagate(int pTime = 0) {
    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int inputTimeSize = input->GetTimeSize();
    int inputColSize = input->GetColSize();

    int batchsize     = result->GetBatchSize();
    int channelsize   = result->GetChannelSize();
    int rowsize       = result->GetRowSize();
    int colsize       = result->GetColSize();

    Shape *inputTenShape  = input->GetShape();
    Shape *resultTenShape = result->GetShape();

    for (int ti = 0; ti < inputTimeSize; ti++){
        for (int ba = 0; ba < batchsize; ba++) {
              if((*input)[Index5D(inputTenShape, ti, ba, 0, 0, 0)] == m_paddingTok)
                  (*result)[Index5D(resultTenShape, 0, ba, 0, 0, ti)] = 1;
              else
                  (*result)[Index5D(resultTenShape, 0, ba, 0, 0, ti)] = 0;
        }
    }

    // std::cout<<"완료"<<'\n';
    // std::cout<<resultTenShape<<'\n';
    // std::cout<<result<<'\n';


    return TRUE;
  }

  //어떠한 연산이 필요가 없음!
  int BackPropagate(int pTime = 0) {

    return TRUE;

  }
};

#endif
