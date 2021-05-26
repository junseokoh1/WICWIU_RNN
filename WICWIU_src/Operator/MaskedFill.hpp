#ifndef __MASKED_FILL_HPP__
#define __MASKED_FILL_HPP__

#include "../Operator.hpp"

template<typename DTYPE> class MaskedFill : public Operator<DTYPE> {
private:
public:
  MaskedFill(Operator<DTYPE> *pInput, Operator<DTYPE> *pMask, std::string pName = "NO NAME", int pLoadflag = TRUE);

  int     Alloc(Operator<DTYPE> *pInput);

  int     ForwardPropagate(int pTime = 0);
  int     BackPropagate(int pTime = 0);
};

template<typename DTYPE> MaskedFill<DTYPE>::MaskedFill(Operator<DTYPE> *pInput, Operator<DTYPE> *pMask, std::string Name, int pLoadflag) : Operator<DTYPE>(pInput, pMask, Name, pLoadflag) {
  #ifdef __DEBUG__
  std::cout << "MaksedFill<DTYPE>::MaskedFill(Operator<DTYPE> *, Operator<DTYPE> *, std::string , int )" << '\n';
  #endif  // __DEBUG__

  Alloc(pInput);
}


template<typename DTYPE> int MaskedFill<DTYPE>::Alloc(Operator<DTYPE> *pInput) {
  #ifdef __DEBUG__
  std::cout << "MaksedFill<DTYPE>::MaskedFill(Operator<DTYPE> * , Operator<DTYPE> * )" << '\n';
  #endif  // __DEBUG__

  Tensor<DTYPE> *pTensor = pInput->GetResult();

  int timesize    = pTensor->GetTimeSize();
  int batchsize   = pTensor->GetBatchSize();
  int channelsize = pTensor->GetChannelSize();
  int rowsize     = pTensor->GetRowSize();
  int colsize     = pTensor->GetColSize();

  this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

  this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

  return TRUE;
};

template<typename DTYPE> int MaskedFill<DTYPE>::ForwardPropagate(int pTime) {

    //std::cout<<"MaskedFill forward 호출"<<'\n';
    //지금 segmentation fault가 뜨는 이유는 mask가 잘못 생성되어 있어서 문제가 있음!!!


    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *mask   = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int maskBatch   = mask->GetBatchSize();

    Shape *resultTenShape = result->GetShape();
    Shape *maskTenShape   = mask->GetShape();

    //mask값 0 : padding 안된거   1 : padding 된 부분
    // std::cout<<"input shape"<<'\n';
    // std::cout<<input->GetShape()<<'\n';
    //
    // std::cout<<"mask shape"<<'\n';
    // std::cout<<mask->GetShape()<<'\n';
    //std::cout<<mask<<'\n';

    int ti = pTime;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ch = 0; ch < channelsize; ch++) {
            for (int ro = 0; ro < rowsize; ro++) {
                for (int co = 0; co < colsize; co++) {
                    int index = Index5D(resultTenShape, ti, ba ,ch, ro ,co);
                    if((*mask)[Index5D(maskTenShape, 0, ba, 0, ro, co)])              //여기만 ti를 0으로 수정함!
                      (*result)[index] = -1e9;
                    else
                      (*result)[index] = (*input)[index];
                }
            }
        }
    }

    // std::cout<<"MaskedFill forward 호출 완료"<<'\n';
    // std::cout<<result<<'\n';

    return TRUE;
}


//padding이 안된 부분은 그냥 그대로 넘겨주면 됨!
//padding된 부분은 gradient가 흐르지 않아야 되니깐 0!
// 별다른 작업을 하지 않아도 위에서 0이 오기 때문에 알아서 0이 됨!!!
//수정해야됨!
//근데 수정을 안해도 작동은 됨!

template<typename DTYPE> int MaskedFill<DTYPE>::BackPropagate(int pTime) {

    Tensor<DTYPE> *mask   = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *this_delta  = this->GetDelta();

    Shape* pThisDeltaTenShape  = this_delta->GetShape();
    Shape* pInputDeltaTenShape = input_delta->GetShape();

    Shape *maskTenShape   = mask->GetShape();

    int timesize    = input_delta->GetTimeSize();
    int batchsize   = input_delta->GetBatchSize();
    int channelsize = input_delta->GetChannelSize();
    int rowsize     = input_delta->GetRowSize();
    int colsize     = input_delta->GetColSize();

    // std::cout<<"maskedfill backward"<<'\n';
    // std::cout<<this_delta<<'\n';

    int ti = pTime;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ch = 0; ch < channelsize; ch++) {
            for (int ro = 0; ro < rowsize; ro++) {
                for (int co = 0; co < colsize; co++) {
                    if((*mask)[Index5D(maskTenShape, 0, ba, 0, ro, co)]) {                  //여기도 ti을 0으로 수정해줌!   ch에 1 -> 0으로 수정...
                      int index = Index5D(pThisDeltaTenShape, ti, ba ,ch, ro ,co);
                      (*input_delta)[index] = (*this_delta)[index] * -1e9;
                    }
                }
            }
        }
    }

    return TRUE;
}

#endif
