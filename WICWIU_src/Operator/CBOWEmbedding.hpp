#ifndef CBOWEMBEDDING_H_
#define CBOWEMBEDDING_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class CBOWEmbedding : public Operator<DTYPE>{
private:

    Operator<DTYPE> *m_aTempInput1;
    Operator<DTYPE> *m_aTempInput2;
    Operator<DTYPE> *m_aInput2Hidden1;
    Operator<DTYPE> *m_aInput2Hidden2;
    Operator<DTYPE> *m_aAddInput;
    Operator<DTYPE> *m_aHidden2Output;

public:

    //인자에 int 넘기는거 불가능!....  operator만 넘겨주는거 가능!
    CBOWEmbedding(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight_in, Operator<DTYPE> *pWeight_out, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pWeight_in, pWeight_out, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "CBOWEmbedding::CBOWEmbedding(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeight_in, pWeight_out);
    }

    virtual ~CBOWEmbedding() {
        #ifdef __DEBUG__
        std::cout << "CBOWEmbedding::~CBOWEmbedding()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight_in, Operator<DTYPE> *pWeight_out) {
        #ifdef __DEBUG__
        std::cout << "CBOWEmbedding::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int vocabsize     = pWeight_in->GetResult()->GetColSize();

        std::cout<<pInput->GetResult()->GetColSize()<<'\n';
        std::cout<<rowsize<<'\n';

        std::cout<<"CBOWEmbedding Alloc"<<'\n';
        std::cout<<"timesize : "<<timesize<<'\n';
        std::cout<<"batchsize : "<<batchsize<<'\n';
        std::cout<<"vocabsize : "<<vocabsize<<'\n';

        m_aTempInput1        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, vocabsize), "tempInput1");
        m_aTempInput2        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, vocabsize), "tempInput2");
        m_aInput2Hidden1     = new Embedding<DTYPE>(pWeight_in, m_aTempInput1, "cbow_embedding_win", TRUE);
        m_aInput2Hidden2     = new Embedding<DTYPE>(pWeight_in, m_aTempInput2, "cbow_embedding_win");
        m_aAddInput          = new Addall<DTYPE>(m_aInput2Hidden1, m_aInput2Hidden2, "cbow_add_input_");
        //나누기 1/2
        m_aHidden2Output  = new MatMul<DTYPE>(pWeight_out, m_aAddInput, "cbow_matmul_wout");

        //For AnalyzeGraph
        pWeight_in->GetOutputContainer()->Pop(m_aInput2Hidden1);
        pWeight_in->GetOutputContainer()->Pop(m_aInput2Hidden2);
        pWeight_out->GetOutputContainer()->Pop(m_aHidden2Output);

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, vocabsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, vocabsize));

        return TRUE;
    }




    void Delete() {

    }

    int ForwardPropagate(int pTime = 0) {

        //입력값 복사해주기
        Operator<DTYPE> *pInput  = this->GetInput()[0];
        Operator<DTYPE> *pWeight_in  = this->GetInput()[1];

        Tensor<DTYPE> *InputTensor     = pInput->GetResult();
        Tensor<DTYPE> *tempInput1   = m_aTempInput1->GetResult();
        Tensor<DTYPE> *tempInput2   = m_aTempInput2->GetResult();

        Shape *inputShape = InputTensor->GetShape();
        Shape *tempInputShape   = tempInput1->GetShape();

        int batchsize = (*inputShape)[1];
        int vocabsize = pWeight_in->GetResult()->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<vocabsize; i++){
                (*tempInput1)[Index5D(tempInputShape, pTime, ba, 0, 0, i)]    = (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, i)];
                (*tempInput2)[Index5D(tempInputShape, pTime, ba, 0, 0, i)]    = (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, vocabsize+i)];
            }
        }

        m_aTempInput1->ForwardPropagate(pTime);
        m_aTempInput2->ForwardPropagate(pTime);
        m_aInput2Hidden1->ForwardPropagate(pTime);
        m_aInput2Hidden2->ForwardPropagate(pTime);
        m_aAddInput->ForwardPropagate(pTime);

        // 2로 나눠주기!       방법1: 접근해서 하기    방법2: matmul 사용.... 방법3: 새로운 operator 만들기

        m_aHidden2Output->ForwardPropagate(pTime);


        //결과값 복사해주기
        Tensor<DTYPE> *_result = m_aHidden2Output->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        Shape *ResultShape = result->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < vocabsize; i++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        std::cout<<"cbowembedding forward 끝"<<'\n';

        return TRUE;
    }



    int BackPropagate(int pTime = 0) {

        //Gradient값 복사해주기
        Tensor<DTYPE> *_grad = m_aHidden2Output->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int batchsize      = grad->GetBatchSize();
        int colSize        = grad->GetColSize();
        Shape *ResultShape = grad->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*_grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        m_aHidden2Output->BackPropagate(pTime);

        //gradient 곱하기 2 해주기
        m_aAddInput->BackPropagate(pTime);
        m_aInput2Hidden2->BackPropagate(pTime);
        m_aInput2Hidden1->BackPropagate(pTime);
        m_aTempInput2->BackPropagate(pTime);
        m_aTempInput1->BackPropagate(pTime);


        //입력으로 gradient값 복사해주기
        Operator<DTYPE> *pInput  = this->GetInput()[0];

        Tensor<DTYPE> *InputTensor      = pInput->GetGradient();
        Tensor<DTYPE> *tempInput1       = m_aTempInput1->GetGradient();
        Tensor<DTYPE> *tempInput2       = m_aTempInput2->GetGradient();

        Shape *inputShape = InputTensor->GetShape();
        Shape *tempInputShape   = tempInput1->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<colSize; i++){
                (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, i)]            = (*tempInput1)[Index5D(tempInputShape, pTime, ba, 0, 0, i)];
                (*InputTensor)[Index5D(inputShape, pTime, ba, 0, 0, colSize+i)]    = (*tempInput2)[Index5D(tempInputShape, pTime, ba, 0, 0, i)];
            }
        }

        return TRUE;
    }

    int ResetResult() {

        m_aTempInput1->ResetResult();
        m_aTempInput2->ResetResult();
        m_aInput2Hidden1->ResetResult();
        m_aInput2Hidden2->ResetResult();
        m_aAddInput->ResetResult();
        m_aHidden2Output->ResetResult();

    }

    int ResetGradient() {
        m_aTempInput1->ResetGradient();
        m_aTempInput2->ResetGradient();
        m_aInput2Hidden1->ResetGradient();
        m_aInput2Hidden2->ResetGradient();
        m_aAddInput->ResetGradient();
        m_aHidden2Output->ResetGradient();
    }
};


#endif  // CBOW_H_
