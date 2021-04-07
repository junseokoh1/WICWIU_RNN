#ifndef NEG_H_
#define NEG_H_    value

#include "../LossFunction.hpp"

/*!
@class NEG Cross Entropy Metric를 이용해 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details Cross Entropy 계산식을 이용해 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와 레이블 값의 손실 함수를 계산한다
*/
template<typename DTYPE>
class NEG : public LossFunction<DTYPE>{
private:
    DTYPE m_epsilon = 1e-6f;  // for backprop

public:

    NEG(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon = 1e-6f) : LossFunction<DTYPE>(pOperator, pLabel) {
        #ifdef __DEBUG__
        std::cout << "NEG::NEG(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }


    NEG(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "NEG::NEG(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, 1e-6f);
    }


    NEG(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, int epsilon, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "NEG::NEG(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }


    ~NEG() {
        #ifdef __DEBUG__
        std::cout << "NEG::~NEG()" << '\n';
        #endif  // __DEBUG__
    }


    virtual int Alloc(Operator<DTYPE> *pOperator, int epsilon) {
        #ifdef __DEBUG__
        std::cout << "NEG::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        // std::cout<<'\n'<<"NEG의 입력 값"<<'\n';
        // std::cout<<input<<'\n';

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = (ti * batchsize + ba);
            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += -log(  (*input)[index] + m_epsilon );

              //  std::cout<<j<<"번째 NEG 계산 결과값 : "<<-log(  (*input)[index] + m_epsilon )<<'\n';
                //
                // if(isnan( -log((*input)[index] + m_epsilon ))){
                //     std::cout<<'\n'<<"NEG Forward에서 nan 발생"<<'\n';
                //     exit(0);
                // }
            }
        }

        //std::cout<<"최종 결과"<<'\n'<<result<<'\n';

        return result;
    }

    Tensor<DTYPE>* BackPropagate(int pTime = 0) {

      //  std::cout<<"--------------------NEG Backpropagate-----------------"<<'\n';

        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            //(*input_delta)[i * capacity] += -1 / (*input)[i * capacity];

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[index] += -1 / ( (*input)[index] + m_epsilon );

              //  std::cout<<j<<"번째 NEG 계산 결과값 : "<<-1 / ( (*input)[index] + m_epsilon )<<'\n';

            }
        }

      //  std::cout<<"끝"<<'\n';

        return NULL;
    }

};

#endif
