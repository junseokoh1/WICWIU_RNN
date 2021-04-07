#ifndef PADDINGATTENTIONMASK_HPP_
#define PADDINGATTENTIONMASK_HPP_

#include "../Operator.hpp"


/*

  입력으로 index를 준다고 생각을 하고 만든거!!!
  그래서 입력을 one-hot으로 주면 이상하게 됨!

*/

template<typename DTYPE> class PaddingAttentionMask : public Operator<DTYPE> {
private:
  Tensor<DTYPE>* m_aSubSequentMask;
  DTYPE m_paddingTok;
public:
  PaddingAttentionMask(Operator<DTYPE> *pInput, int vocabLength, DTYPE mask = 0.F, int IsDecoder = FALSE, std::string Name = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput, Name, pLoadflag) {
    #ifdef __DEBUG__
    std::cout << "PaddingAttentionMask<DTYPE>::PaddingAttentionMask(Operator<DTYPE> , int , int , int , std::string , int )" << '\n';
    #endif  // __DEBUG__

    m_aSubSequentMask = NULL;
    m_paddingTok = 0;

    Alloc(pInput, vocabLength, mask, IsDecoder);
  }

  int Alloc(Operator<DTYPE> *pInput, int vocabLength, int mask, int IsDecoder) {
    #ifdef __DEBUG__
    std::cout << "PaddingAttentionMask<DTYPE>::Alloc(Operator<DTYPE> *, int " << '\n';
    #endif  // __DEBUG__

    m_paddingTok = mask;

    int timesize    = pInput->GetResult()->GetTimeSize();
    int batchsize   = pInput->GetResult()->GetBatchSize();
    int channelsize = pInput->GetResult()->GetChannelSize();
    int colsize     = pInput->GetResult()->GetColSize();
    int rowsize     = colsize;


    this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
    this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

    m_aSubSequentMask = Tensor<DTYPE>::Constants(1, 1, 1, colsize, colsize, 1);

    // if(IsDecoder)
    //   m_aSubSequentMask->TriangleLower(0, 0.F);
  }


  int ForwardPropagate(int pTime = 0) {
    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    Shape *inputTenShape  = input->GetShape();
    Shape *resultTenShape = result->GetShape();
    Shape *subsequentMaskTenShape = m_aSubSequentMask->GetShape();

    int ti = pTime;

    DTYPE fill;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ch = 0; ch < channelsize; ch++) {
            for (int co = 0; co < colsize; co++) {
                if((*input)[Index5D(inputTenShape, ti, ba, ch, 0, co)] == m_paddingTok)
                  fill = 0.F;
                else
                  fill = 1.F;

                for (int ro = 0; ro < rowsize; ro++) {
                    (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] = 1-fill*(*m_aSubSequentMask)[Index5D(subsequentMaskTenShape, 0, 0, 0, ro, co)];
                }
            }
        }
    }


    std::cout<<"완료"<<'\n';


    return TRUE;
  }

  int BackPropagate(int pTime = 0) {
    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    Shape *inputTenShape  = input->GetShape();
    Shape *resultTenShape = result->GetShape();

    int ti = pTime;


    return TRUE;

  }
};

#endif
