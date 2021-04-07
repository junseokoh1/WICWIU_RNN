#ifndef BASERNN_H_
#define BASERNN_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

template<typename DTYPE> class BaseRNN : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *m_aPostActivate;
    Operator<DTYPE> *m_aPrevActivateBias;

public:
    BaseRNN(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) : Operator<DTYPE>(4, pInput, pWeightXH, pWeightHH, rBias) {
        this->Alloc(pInput, pWeightXH, pWeightHH, rBias);
    }

    BaseRNN(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName) : Operator<DTYPE>(4, pInput, pWeightXH, pWeightHH, rBias, pName) {
        this->Alloc(pInput, pWeightXH, pWeightHH, rBias);
    }

    ~BaseRNN() {
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightXH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[0];
        int hidBatchSize = (*InputShape)[1];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightXH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");

        m_aPrevActivateBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");

        m_aPostActivate  = new Tanh<DTYPE>(m_aPrevActivateBias, "rnn_tanh");

        rBias->GetOutputContainer()->Pop(m_aPrevActivateBias);
        pWeightXH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);


        Shape *ResultShape = m_aPostActivate->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[0];
        int batchSize = (*ResultShape)[1];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));      //Container<Tensor<DTYPE> *> *m_aaResult;
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));    //Container<Tensor<DTYPE> *> *m_aaGradient;

        return TRUE;
    }

    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        m_aInput2Hidden->ForwardPropagate(pTime);


        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = m_aPostActivate->GetResult();                  //m_aHidden2Hidden이 아니라 aPostActivate로 바꿔야 될 꺼 같음!!!   바꾸면 m_aTempHidden선언할 때 사이즈도 확인해줄것!!!  바
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];        //pTime-1 이게 핵심임!!!
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
        }
        m_aPrevActivate->ForwardPropagate(pTime);                               //time=0 일때는 hidden에서 hidden으로 가는게 필요없으니깐 바로 여기로

        m_aPrevActivateBias->ForwardPropagate(pTime);

        m_aPostActivate->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = m_aPostActivate->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }


        return TRUE;
    }

    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = m_aPostActivate->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();


        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();

        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }


        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aPostActivate->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }

        m_aPostActivate->BackPropagate(pTime);

        m_aPrevActivateBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        return TRUE;
    }




    int ResetResult() {
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        m_aPostActivate->ResetResult();
      //  m_aHidden2Output->ResetResult();
        m_aPrevActivateBias->ResetResult();
    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        m_aPostActivate->ResetGradient();
        //m_aHidden2Output->ResetGradient();
        m_aPrevActivateBias->ResetGradient();
    }


};


#endif  // RECURRENT_H_
