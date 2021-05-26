#ifndef FLIP_H_
#define FLIP_H_    value

#include "../Operator.hpp"

//이것도 결과 확인!
//attetnion 때문에 추가
template<typename DTYPE>
class FlipTimeWise : public Operator<DTYPE>{
private:

public:
    FlipTimeWise(Operator<DTYPE> *pInput0, std::string pName = "NO NAME", int pLoadflag = TRUE) : Operator<DTYPE>(pInput0, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "FlipTimeWise::FlipTimeWise(Operator *)" << '\n';
        #endif  // __DEBUG__

        this->Alloc(pInput0);
    }

    ~FlipTimeWise() {
        std::cout << "FlipTimeWise::~FlipTimeWise()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput0) {
        #ifdef __DEBUG__
        std::cout << "FlipTimeWise::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__


        int timesize    = pInput0->GetResult()->GetTimeSize();
        int batchsize   = pInput0->GetResult()->GetBatchSize();
        int channelsize = pInput0->GetResult()->GetChannelSize();
        int rowsize     = pInput0->GetResult()->GetRowSize();
        int colsize     = pInput0->GetResult()->GetColSize();


        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ForwardPropagate(int pTime = 0) {

        //std::cout<<"ConcatenateColumnWise Forward 호출 "<<pTime<<'\n';
        if(pTime !=0)
          return TRUE;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize     = result->GetTimeSize();
        int batchsize    = result->GetBatchSize();
        int channelsize  = result->GetChannelSize();
        int rowsize      = result->GetRowSize();
        int colsize      = result->GetColSize();

        Shape *inputTenShape = input->GetShape();
        Shape *resultTenShape = result->GetShape();


        for(int ti=0; ti < timesize; ti++){
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                = (*input)[Index5D(inputTenShape, timesize - ti - 1, ba, ch, ro, co)];
                        }
                    }
                }
            }
        }



        return TRUE;
    }

    int BackPropagate(int pTime = 0) {

        //std::cout<<"ConcatenateColumnWise Forward 호출 "<<pTime<<'\n';

        Tensor<DTYPE> *inputGradient  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *thisGradient = this->GetGradient();

        int timesize     = inputGradient->GetTimeSize();
        int batchsize    = inputGradient->GetBatchSize();
        int channelsize  = inputGradient->GetChannelSize();
        int rowsize      = inputGradient->GetRowSize();
        int colsize      = inputGradient->GetColSize();

        if(pTime != timesize-1)
          return TRUE;

        Shape *inputTenShape = inputGradient->GetShape();
        Shape *resultTenShape = thisGradient->GetShape();


        for(int ti=0; ti < timesize; ti++){
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*inputGradient)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                                = (*thisGradient)[Index5D(resultTenShape, timesize - ti - 1, ba, ch, ro, co)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__
};




#endif  // FLIP_H_
