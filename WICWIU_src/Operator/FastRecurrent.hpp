#ifndef FASTRECURRENT_H_
#define FASTRECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1


//속도를 빠르게 하기 위해!!! 수정!

template<typename DTYPE> class FastRecurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;

    //initHidden
    Operator<DTYPE> *m_aInit2Hidden;
    Operator<DTYPE> *m_aInitHidden;


#ifdef __CUDNN__
    //cudnnRNNDescriptor_t rnnDesc;
    //  cudnnRNNDataDescriptor_t RNNDataDesc;
    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
#endif  // __CUDNN__

public:

    //Sequence to Sequence 때문에 추가!
    //이거 Operator 생성자 호출할 때 굳이 숫자 4 -> 5로 안바꾸는 이유!!!.... 굳이 연결이 필요없지 않을까?....
    FastRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "SeqRecurrent::SeqRecurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pInitHidden);
    }

    //pName때문에 Operator 생성자 호출이 안되는듯!!!!   숫자 4로해도 되는건가?
    FastRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "SeqRecurrent::SeqRecurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pInitHidden);
    }

    ~FastRecurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pInitHidden) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aHidden2Hidden = new HiddenMatMul<DTYPE>(pWeightHH, ApplyActivation, "rnn_matmul_hh");                      //그래프 안 꼬이는지 확인하기!!!
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        AddBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");
        ApplyActivation  = new Tanh<DTYPE>(AddBias, "rnn_tanh");
        //ApplyActivation  = new Relu<DTYPE>(AddBias, "rnn_tanh");

        //for initHidden
        m_aInit2Hidden = new MatMul<DTYPE>(pWeightHH, pInitHidden, "ddd");                     //더하는걸 m_aPrevActivate에 time=0 일때만 더하기... operator에서 내가 임의로!
        m_aInitHidden = pInitHidden;

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aInput2Hidden);   // 21년 3월 4일 추가!
        rBias->GetOutputContainer()->Pop(AddBias);
        pWeightIH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aInit2Hidden);

        Shape *ResultShape = ApplyActivation->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }


#if __CUDNN__
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

          m_alpha = 1;
          m_beta  = 0;

          m_aInput2Hidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aTempHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aHidden2Hidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          m_aPrevActivate->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          AddBias->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          ApplyActivation->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

          //checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc));
          //checkCUDNN(cudnnCreateRNNDataDescriptor(&RNNDataDesc));
      }

#endif  // if __CUDNN__


    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        m_aInput2Hidden->ForwardPropagate(pTime);

        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime==0 && m_aInitHidden != NULL){


        }
        //*********************************************inithidden 처리하기!!!*****************************************

        if(pTime != 0)
          m_aHidden2Hidden->ForwardPropagate(pTime);

        m_aPrevActivate->ForwardPropagate(pTime);

        AddBias->ForwardPropagate(pTime);


        ApplyActivation->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        int batchsize      = result->GetBatchSize();
        Shape *ResultShape = result->GetShape();

        for(int ba = 0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }



        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        // std::cout<<pTime<<" seqRecurrent Backward 호출"<<'\n';

        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        int batchsize      = grad->GetBatchSize();
        Shape *ResultShape = grad->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*_grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        // std::cout<<grad->GetShape()<<'\n';
        // std::cout<<grad<<'\n';

        // if(pTime == timeSize-1){
        //     std::cout<<pTime<<"  seqRecurrent backward 호출"<<'\n';
        //     std::cout<<grad<<'\n';
        // }


        //std::cout<<_grad<<'\n';

        if (pTime != timeSize-1)
            m_aHidden2Hidden->BackPropagate(pTime);


        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime == 0 && m_aInitHidden != NULL){

        }
        //*********************************************inithidden 처리하기!!!*****************************************

        return TRUE;
    }





    // GPU에 대한 Reset 처리는 operator.hpp에 되어있음
    int ResetResult() {
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aPrevActivate->ResetResult();
        ApplyActivation->ResetResult();
        AddBias->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        if(m_aInitHidden != NULL)
           m_aInitHidden->GetResult();

    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        ApplyActivation->ResetGradient();
        AddBias->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

        if(m_aInitHidden != NULL)
          m_aInitHidden->ResetGradient();

    }


};


#endif  // SEQRECURRENT_H_
