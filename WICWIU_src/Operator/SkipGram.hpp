#ifndef SKIPGRAM_H_
#define SKIPGRAM_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class SkipGram : public Operator<DTYPE>{
private:

    Operator<DTYPE> *m_aCenterIndex;
    Operator<DTYPE> *m_aNeiborIndex;

    Operator<DTYPE> *m_aCenter;
    Operator<DTYPE> *m_aNeibor;

    Operator<DTYPE> *m_aOutput;


public:

    SkipGram(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight_in, Operator<DTYPE> *pWeight_out, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight_in, pWeight_out, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "SkipGram::SkipGram(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeight_in, pWeight_out);
    }

    virtual ~SkipGram() {
        #ifdef __DEBUG__
        std::cout << "SkipGram::~SkipGram()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight_in, Operator<DTYPE> *pWeight_out) {
        #ifdef __DEBUG__
        std::cout << "SkipGram::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize      = pInput->GetResult()->GetTimeSize();
        int batchsize     = pInput->GetResult()->GetBatchSize();
        int channelsize   = pInput->GetResult()->GetChannelSize();
        int rowsize       = pInput->GetResult()->GetRowSize();
        int colsize       = pInput->GetResult()->GetColSize();

        m_aCenterIndex        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, 1), "centerIndex");
        m_aNeiborIndex        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, colsize-1), "neiborIndex");

        m_aCenter             = new Embedding<DTYPE>(pWeight_in, m_aCenterIndex, "SkipGram_target_embed");
        m_aNeibor             = new Embedding<DTYPE>(pWeight_out, m_aNeiborIndex, "SkipGram_positive_embed");

        //m_aOutput             = new MatMul<DTYPE>(m_aCenter, m_aNeibor, "SkipGram_matmul");     //이렇게 하면 shape : 1 X 1 X 1 X 4 X 1
        //m_aOutput             = new MatMul<DTYPE>(m_aNeibor, m_aCenter, "SkipGram_matmul");       //이렇게 하면 shape : 1 X 1 X 1 X 1 X 4

        m_aOutput             = new DotProduct<DTYPE>(m_aNeibor, m_aCenter, "SkipGram_DotProduct");       //이렇게 하면 shape : 1 X 1 X 1 X 1 X 4

        //For AnalyzeGraph
        pWeight_in->GetOutputContainer()->Pop(m_aCenter);
        pWeight_out->GetOutputContainer()->Pop(m_aNeibor);


        //target에 대한게 하나 사라져서 그래서 -1을 해줌!!!
        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize-1));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize-1));

        // this->SetResult(Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize-1));
        // this->SetGradient(Tensor<DTYPE>::Zeros(timesize, batchsize, channelsize, rowsize, colsize-1));

// //-------------test를 위해 추가!!! 여기 이후에 있는건 test에서만 사용!!!-------------------------------
//         m_aTestIndex         = new Embedding<DTYPE>(pWeight_in, pInput, "SkipGram_test_index");
//
//         //For AnalyzeGraph
//         pWeight_in->GetOutputContainer()->Pop(m_aTestIndex);
//
//     //방법2 weight를 복사해서 해보자!!!
//         //
//         // m_aTestWeight = new Tensorholder<DTYPE>(pWeight_in->GetTensor(), "testWeight");       //지금 여기서 복사해봤자... 학습전의 weiht를 복사해 가는거임!!!
//         flag = 0;

        return TRUE;
    }




    void Delete() {

    }

    int ForwardPropagate(int pTime = 0) {

        //입력값 복사해주기
        Operator<DTYPE> *pInput  = this->GetInput()[0];

        Tensor<DTYPE> *InputTensor          = pInput->GetResult();
        Tensor<DTYPE> *tempCenterInput      = m_aCenterIndex->GetResult();
        Tensor<DTYPE> *tempNeiborInput      = m_aNeiborIndex->GetResult();


        Shape *inputShape          = InputTensor->GetShape();
        Shape *tempCenterShape     = tempCenterInput->GetShape();
        Shape *tempNeiborShape     = tempNeiborInput->GetShape();

        int batchsize  =  (*inputShape)[1];
        int colsize    =  (*inputShape)[4];

        for(int ba=0; ba<batchsize; ba++){
            //center word Index!!!
            (*tempCenterInput)[Index5D(tempCenterShape, pTime, ba, 0, 0, 0)]  =  (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, 0)];
            //neibor words Index
            for(int col=1; col < colsize; col++){
                  (*tempNeiborInput)[Index5D(tempNeiborShape, pTime, ba, 0, 0, col-1)] = (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, col)];
            }
        }

        // std::cout<<'\n'<<"skipgram operator에서 입력값 확인하기"<<'\n';
        // std::cout<<"center input"<<'\n';
        // std::cout<<m_aCenterIndex->GetResult()<<'\n';
        // std::cout<<"Neibor input"<<'\n';
        // std::cout<<m_aNeiborIndex->GetResult()<<'\n';

        m_aCenter->ForwardPropagate(pTime);
        m_aNeibor->ForwardPropagate(pTime);

        // std::cout<<'\n'<<"center shape"<<m_aCenter->GetResult()->GetShape()<<'\n';
        // std::cout<<'\n'<<"m_aNeibor shape"<<m_aNeibor->GetResult()->GetShape()<<'\n';

        // std::cout<<"matmul 전 결과들"<<'\n'<<"m_aCenter"<<'\n';
        // std::cout<<m_aCenter->GetResult()<<'\n';
        // std::cout<<"m_aNeibor"<<'\n'<<m_aNeibor->GetResult()<<'\n';

        m_aOutput->ForwardPropagate(pTime);

        // std::cout<<'\n'<<"m_aOutput shape"<<m_aOutput->GetResult()->GetShape()<<'\n';

        //std::cout<<'\n'<<"내적 결과"<<m_aOutput->GetResult()<<'\n';                              //이걸로 계속 출력해보고 있었음!!!

        //결과값 복사해주기
        Tensor<DTYPE> *_result = m_aOutput->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        Shape *ResultShape = result->GetShape();
        Shape *tempShape = _result->GetShape();

        colsize = (*ResultShape)[4];

        for(int ba=0; ba<batchsize; ba++){
            //positive sample
            (*result)[Index5D(ResultShape, pTime, ba, 0, 0, 0)] = (*_result)[Index5D(tempShape, pTime, ba, 0, 0, 0)];
            //negative sample  !중요! 실제 값을 변경시켜주는것도 중요함!!!
            for(int col=1; col<colsize; col++){
                (*_result)[Index5D(tempShape, pTime, ba, 0, 0, col)] = -(*_result)[Index5D(tempShape, pTime, ba, 0, 0, col)];                           // 여기 한줄!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, col)]  = (*_result)[Index5D(tempShape, pTime, ba, 0, 0, col)];
            }
        }

        #ifdef __SKIPGRAM__
        std::cout<<'\n'<<"skipgram forward 결과"<<'\n'<<result<<'\n';
        #endif

        return TRUE;
    }



    int BackPropagate(int pTime = 0) {

        //std::cout<<"---------------------------skipGram backpropagate------------------"<<'\n';

        //Gradient값 복사해주기
        Tensor<DTYPE> *_grad = m_aOutput->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int batchsize      = grad->GetBatchSize();
        int colsize        = grad->GetColSize();
        Shape *ResultShape = grad->GetShape();
        Shape *tempShape = _grad->GetShape();

        // std::cout<<ResultShape<<'\n';
        // std::cout<<grad<<'\n'<<colsize<<'\n';

        for(int ba=0; ba<batchsize; ba++){
             //positive sample
             (*_grad)[Index5D(tempShape, pTime, ba, 0, 0, 0)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, 0)];
             //negative sample
             for (int col = 1; col < colsize; col++){
                //실제 값을 바꾸려고 한줄 추가한거!!!
                (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, col)] = -(*grad)[Index5D(ResultShape, pTime, ba, 0, 0, col)];                         //여기도 한줄!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                (*_grad)[Index5D(tempShape, pTime, ba, 0, 0, col)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, col)];
            }
        }

        #ifdef __SKIPGRAM__
        std::cout<<"skip gram에서 받은 gradient"<<'\n';
        std::cout<<'\n'<<_grad<<'\n';
        #endif

        m_aOutput->BackPropagate(pTime);
        m_aNeibor->BackPropagate(pTime);
        m_aCenter->BackPropagate(pTime);

        //입력으로 gradient값 복사해주기
        Operator<DTYPE> *pInput  = this->GetInput()[0];

        Tensor<DTYPE> *InputTensor          = pInput->GetGradient();
        Tensor<DTYPE> *tempCenterInput      = m_aCenterIndex->GetGradient();
        Tensor<DTYPE> *tempNeiborInput      = m_aNeiborIndex->GetGradient();

        Shape *inputShape       = InputTensor->GetShape();
        Shape *tempCenterShape     = tempCenterInput->GetShape();
        Shape *tempNeiborShape     = tempNeiborInput->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            //center word Index!!!
            (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, 0)]  =  (*tempCenterInput)[Index5D(tempCenterShape, pTime, ba, 0, 0, 0)];
            //neibor words Index
            for(int col=1; col < colsize; col++)
                  (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, col)] = (*tempNeiborInput)[Index5D(tempNeiborShape, pTime, ba, 0, 0, col-1)];
        }

/*
        std::cout<<'\n'<<"Skipgram back의 최종 결과!"<<'\n';
        std::cout<<InputTensor<<'\n';
        std::cout<<tempCenterInput<<'\n';
        std::cout<<tempNeiborInput<<'\n';
*/
        return TRUE;
    }


    int ResetResult() {

        m_aCenterIndex->ResetResult();
        m_aNeiborIndex->ResetResult();
        m_aCenter->ResetResult();
        m_aNeibor->ResetResult();
        m_aOutput->ResetResult();

       Tensor<DTYPE> *result  = this->GetResult();       //이게 문제인가?.... 이게 문제라고 가정하고 돌려보자!
       result->Reset();
    }

    int ResetGradient() {

        m_aCenterIndex->ResetGradient();
        m_aNeiborIndex->ResetGradient();
        m_aCenter->ResetGradient();
        m_aNeibor->ResetGradient();
        m_aOutput->ResetGradient();

       Tensor<DTYPE> *grad  = this->GetGradient();
       grad->Reset();
    }
};


#endif  // SkipGram_H_
