#ifndef ATTENTIONRECURRENT_H_
#define ATTENTIONRECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class AttentionRecurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;

    //initHidden
    Operator<DTYPE> *m_aInitHidden;
    Operator<DTYPE> *m_attention;             // 이게 추가됨!


#ifdef __CUDNN__
    //cudnnRNNDescriptor_t rnnDesc;
    //  cudnnRNNDataDescriptor_t RNNDataDesc;
    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
#endif  // __CUDNN__

public:

    //attention은 이제 backPropagate하려면 그래프 연결해줘야 된다고 판단됨!!!
    //아니다... 정확히 모르겠다....   정말로 모르겠다!... 나중에 다시 생각해보자
    AttentionRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pAttention, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "AttentionRecurrent::AttentionRecurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pAttention, pInitHidden);
    }

    AttentionRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pAttention, std::string pName, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "AttentionRecurrent::AttentionRecurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pAttention, pInitHidden);
    }

    ~AttentionRecurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pAttention, Operator<DTYPE>* pInitHidden) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_attention      = pAttention
        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        temptemp         = new ConcatenateColWise<DTYPE>(m_attention, m_aTempHidden);
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, temptemp, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        AddBias          = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");
        ApplyActivation  = new Tanh<DTYPE>(AddBias, "rnn_tanh");
        //ApplyActivation  = new Relu<DTYPE>(AddBias, "rnn_tanh");

        //for initHidden
        m_aInitHidden = pInitHidden;

        //For AnalyzeGraph
        rBias->GetOutputContainer()->Pop(AddBias);
        pWeightIH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);

        Shape *ResultShape = ApplyActivation->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }


    //이거 해줘야되나?
    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        m_aInput2Hidden->ForwardPropagate(pTime);

        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime==0 && m_aInitHidden != NULL){

            Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colsize      = initHidden->GetColSize();
            int batchsize    = initHidden->GetBatchSize();
            //2개의 shape은 다르지!!!
            Shape *initShape = initHidden->GetShape();
            Shape *tempShape = tempHidden->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int co = 0; co < colsize; co++){
                    (*tempHidden)[Index5D(tempShape, pTime, ba, 0, 0, co)] = (*initHidden)[Index5D(initShape, ba, 0, 0, 0, co)];
                }
            }

            std::cout<<"inithidden값!"<<'\n';
            std::cout<<initHidden<<'\n';
        }
        //*********************************************inithidden 처리하기!!!*****************************************

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            Shape *HiddenShape = prevHidden->GetShape();

            int batchsize    = tempHidden->GetBatchSize();
            int colsize      = tempHidden->GetColSize();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colsize; i++){
                    (*tempHidden)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, ba, 0, 0, i)];
                }
           }

        }

        m_aHidden2Hidden->ForwardPropagate(pTime);        //time=0일떄 결과가 어차피 0이여서 영향을 안준다고 판단되어서 밖으로 뻄!!!

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

        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevHiddenGrad)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime == 0 && m_aInitHidden != NULL){

            m_aHidden2Hidden->BackPropagate(pTime);

            Tensor<DTYPE> *initHiddenGrad = m_aInitHidden->GetGradient();
            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();

            //2개 shape이 다름 !!! time이 다르게 되어있음!!!
            int colsize      = initHiddenGrad->GetColSize();
            int batchsize    = initHiddenGrad->GetBatchSize();
            Shape *initShape = initHiddenGrad->GetShape();
            Shape *tempShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for(int co=0; co<colsize; co++){
                    (*initHiddenGrad)[Index5D(initShape, 0, ba, 0, 0, co)] += (*tempHiddenGrad)[Index5D(tempShape, pTime, ba, 0, 0, co)];
                  }
            }
        }
        //*********************************************inithidden 처리하기!!!*****************************************

        return TRUE;
    }


    // GPU에 대한 Reset 처리는 operator.hpp에 되어있음
    int ResetResult() {
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        ApplyActivation->ResetResult();
        AddBias->ResetResult();
    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        ApplyActivation->ResetGradient();
        AddBias->ResetGradient();
    }


};


#endif  // ATTENTIONRECURRENT_H_
