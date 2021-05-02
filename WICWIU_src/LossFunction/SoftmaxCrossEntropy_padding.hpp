#ifndef SOFTMAXCROSSENTROPYPADDING_H_
#define SOFTMAXCROSSENTROPYPADDING_H_    value

#include "../LossFunction.hpp"



// 기존의 SCE랑 호환으로 만들지 않음!!!
// 기존의 방식으로도 작동하기 위해서는 if문으로 추가하기! 함!!!
// test 해봐야함!

template<typename DTYPE>
class SoftmaxCrossEntropy_padding : public LossFunction<DTYPE>{
private:
    Tensor<DTYPE> *m_aSoftmaxResult;
    ///< Softmax 연산의 Result 텐서에 대한 포인터
    DTYPE m_epsilon;  // for backprop
    ///< translation 요소 멤버 변수

    int m_Timesize;
    ///< Time 축의 사이즈 멤버 변수

    DTYPE **sum;
    ///< 텐서의 합을 저장하기 위한 이중 포인터
    DTYPE **max;
    ///< 텐서의 최댓값을 저장하기 위한 이중 포인터

    Operator<DTYPE> *m_PaddingLengths;
    //padding을 위해 내가 추가함!!!

public:

    SoftmaxCrossEntropy_padding(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE epsilon, Operator<DTYPE> *pPaddingLengths = NULL, std::string pName = "NO NAME") : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon, pPaddingLengths);
    }


    SoftmaxCrossEntropy_padding(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, Operator<DTYPE> *pPaddingLengths = NULL, std::string pName = "NO NAME") : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, 1e-6f, pPaddingLengths);
    }


    virtual ~SoftmaxCrossEntropy_padding() {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon, Operator<DTYPE> *pPaddingLengths) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_Timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        m_aSoftmaxResult = new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize);

        m_epsilon = epsilon;

        m_PaddingLengths = pPaddingLengths;    //내가 추가한거!

        return TRUE;
    }

    #ifdef __CUDNN__

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_aSoftmaxResult->SetDeviceGPU(idOfDevice);
    }

    #endif  // if __CUDNN__


    virtual void Delete() {
        if (m_aSoftmaxResult) {
            delete m_aSoftmaxResult;
            m_aSoftmaxResult = NULL;
        }

        if (sum) {
            for (int i = 0; i < m_Timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_Timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {

        #if __LOSS__
        std::cout<<'\n'<<"softmaxcrossentropy forward 호출 : "<<pTime<<'\n';
        #endif

        Tensor<DTYPE> *Lengths = m_PaddingLengths->GetResult();

        Tensor<DTYPE> *input         = this->GetTensor();
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

         #if __LOSS__
        if(pTime == 8)
        std::cout<<"softmaxcrossentropy 의 입력값 : "<<'\n'<<input<<'\n';
        std::cout<<"softmaxcrossentropy 의 label 값 : "<<label<<'\n';
        #endif

        for (int ba = 0; ba < batchsize; ba++) {  // thread
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }


        int capacity = colsize;         // colsize는 embedding dim이여서 padding이랑 상관이 없음!

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {

            //padding 처리
            if((*Lengths)[ba] <= ti)
                continue;

            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {

          if((*Lengths)[ba] <= ti)
              continue;

            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < batchsize; ba++) {

          if((*Lengths)[ba] <= ti)                                              //여기서 한번만 해도 처리가 될거 같은데!!! 한번 해보고 된다면 위에 if문 2개는 삭제하기!!!
              continue;

            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*softmaxresult)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];           //여기서 softmax결과 값을 저장해주고

                (*result)[ti * batchsize + ba] += -(*label)[i] * log((*softmaxresult)[i] + m_epsilon);
            }
        }
        #if __LOSS__
        std::cout<<"SoftmaxCrossEntropy forward 결과값(loss값) time : "<<pTime<<'\n'<<result<<'\n';
        #endif

        // std::cout<<Lengths<<'\n';
        // std::cout<<"softmax forward 결과 time "<<pTime<<'\n';
        // std::cout<<result->GetShape()<<'\n';
        // std::cout<<result<<'\n';

        return result;
    }

    /*!
    @brief Softmax CrossEntropy LossFunction의 역전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 Softmax CrossEntropy LossFunction에 대한 입력 Tensor의 Gradient를 계산한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return NULL
    */
    Tensor<DTYPE>* BackPropagate(int pTime = 0) {

        #if __RNNDEBUG__
        std::cout<<"=============================================================softmaxcrossentropy backpropagate 호출 : "<<pTime<<"=============================================================="<<'\n';
        #endif

        Tensor<DTYPE> *Lengths = m_PaddingLengths->GetResult();

        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;

        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input_delta->GetBatchSize();
        int colsize   = input_delta->GetColSize();

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {

          if((*Lengths)[ba] <= ti)
              continue;

            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = ((*softmaxresult)[i] - (*label)[i]);

                #if __RNNDEBUG__
                std::cout<<(*input_delta)[i]<<" = "<<(*softmaxresult)[i]<<" - "<<(*label)[i]<<'\n';
                #endif
            }
        }

        //std::cout<<"연결되어 있는 operator의 이름"<<this->GetOperator()->GetName()<<'\n';  //결과 : Recur_1

        //if(pTime == 0)
        #if __RNNDEBUG__
          std::cout<<"softmaxcrossentropy BackPropagate에서 입력에 해당하는 operator에 넘겨주는 delta : "<<'\n'<<input_delta<<'\n';
        #endif

        return NULL;
    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0);


    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0);


#endif  // __CUDNN__

    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }
};

#endif  // SOFTMAXCROSSENTROPY_H_




//아래는 batch size 방식으로 padding 처리하도록 만든 거!!!!


/*

#ifndef SOFTMAXCROSSENTROPYPADDING_H_
#define SOFTMAXCROSSENTROPYPADDING_H_    value

#include "../LossFunction.hpp"



// 기존의 SCE랑 호환으로 만들지 않음!!!
// 기존의 방식으로도 작동하기 위해서는 if문으로 추가하기! 함!!!
// test 해봐야함!

template<typename DTYPE>
class SoftmaxCrossEntropy_padding : public LossFunction<DTYPE>{
private:
    Tensor<DTYPE> *m_aSoftmaxResult;
    ///< Softmax 연산의 Result 텐서에 대한 포인터
    DTYPE m_Epsilon;  // for backprop
    ///< translation 요소 멤버 변수

    int m_Timesize;
    ///< Time 축의 사이즈 멤버 변수

    DTYPE **sum;
    ///< 텐서의 합을 저장하기 위한 이중 포인터
    DTYPE **max;
    ///< 텐서의 최댓값을 저장하기 위한 이중 포인터

    Operator<DTYPE> *m_PaddingBatchSizes;                  //batch_size에 해당하는 정보를 넘겨준다고 생각!
    //padding을 위해 내가 추가함!!!

public:

    SoftmaxCrossEntropy_padding(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE epsilon, std::string pName = "NO NAME", Operator<DTYPE> *pPaddingBatchSizes = NULL) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon, pPaddingBatchSizes);
    }


    SoftmaxCrossEntropy_padding(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName = "NO NAME", Operator<DTYPE> *pPaddingBatchSizes = NULL) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::SoftmaxCrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, 1e-6f, pPaddingBatchSizes);
    }


    virtual ~SoftmaxCrossEntropy_padding() {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::~SoftmaxCrossEntropy()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon, Operator<DTYPE> *pPaddingBatchSizes) {
        #ifdef __DEBUG__
        std::cout << "SoftmaxCrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_Timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        m_aSoftmaxResult = new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize);

        m_Epsilon = epsilon;

        m_PaddingBatchSizes = pPaddingBatchSizes;    //내가 추가한거!

        return TRUE;
    }

    #ifdef __CUDNN__

    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        m_aSoftmaxResult->SetDeviceGPU(idOfDevice);
    }

    #endif  // if __CUDNN__


    virtual void Delete() {
        if (m_aSoftmaxResult) {
            delete m_aSoftmaxResult;
            m_aSoftmaxResult = NULL;
        }

        if (sum) {
            for (int i = 0; i < m_Timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_Timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {

  //      #if __LOSS__
        std::cout<<'\n'<<"softmaxcrossentropy forward 호출 : "<<pTime<<'\n';
//        #endif

        Tensor<DTYPE> *input         = this->GetTensor();
        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
        Tensor<DTYPE> *result        = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        //padding을 위해 내가 추가!
        int packedBatchsize;
        if(m_PaddingBatchSizes != NULL){
          Tensor<DTYPE> *paddingBatch = m_PaddingBatchSizes->GetResult();
          Shape *paddingShape = paddingBatch->GetShape();
          packedBatchsize = (*paddingBatch)[Index5D(paddingShape, 0, 0, 0, 0, ti)];
        }
        else{
          packedBatchsize = batchsize;
        }



        #if __LOSS__
        std::cout<<"softmaxcrossentropy 의 입력값 : "<<'\n'<<input<<'\n';
        std::cout<<"softmaxcrossentropy 의 label 값 : "<<label<<'\n';
        #endif

        for (int ba = 0; ba < packedBatchsize; ba++) {  // thread
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }


        int capacity = colsize;         //여기가.... padding을 추가하면 핵심!....

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < packedBatchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < packedBatchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_Epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < packedBatchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*softmaxresult)[i] = (exp((*input)[i] - max[ti][ba]) + m_Epsilon) / sum[ti][ba];           //여기서 softmax결과 값을 저장해주고

                (*result)[ti * batchsize + ba] += -(*label)[i] * log((*softmaxresult)[i] + m_Epsilon);
                #if __RNNDEBUG__
                std::cout<<(*label)[i]<<" * "<<log((*softmaxresult)[i] + m_Epsilon)<<'\n';
                #endif
            }
        }
        #if __LOSS__
        std::cout<<"SoftmaxCrossEntropy forward 결과값(loss값) time : "<<pTime<<'\n'<<result<<'\n';
        #endif

        std::cout<<result->GetShape()<<'\n';
        std::cout<<result<<'\n';

        return result;
    }


    Tensor<DTYPE>* BackPropagate(int pTime = 0) {

        #if __RNNDEBUG__
        std::cout<<"=============================================================softmaxcrossentropy backpropagate 호출 : "<<pTime<<"=============================================================="<<'\n';
        #endif

        Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
        Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;

        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input_delta->GetBatchSize();
        int colsize   = input_delta->GetColSize();

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        int ti = pTime;

        //padding을 위해 내가 추가!
        int packedBatchsize;
        if(m_PaddingBatchSizes != NULL){
          Tensor<DTYPE> *paddingBatch = m_PaddingBatchSizes->GetResult();
          Shape *paddingShape = paddingBatch->GetShape();
          packedBatchsize = (*paddingBatch)[Index5D(paddingShape, 0, 0, 0, 0, ti)];
        }
        else{
          packedBatchsize = batchsize;
        }


        for (int ba = 0; ba < packedBatchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = ((*softmaxresult)[i] - (*label)[i]);

                #if __RNNDEBUG__
                std::cout<<(*input_delta)[i]<<" = "<<(*softmaxresult)[i]<<" - "<<(*label)[i]<<'\n';
                #endif
            }
        }

        //std::cout<<"연결되어 있는 operator의 이름"<<this->GetOperator()->GetName()<<'\n';  //결과 : Recur_1

        #if __RNNDEBUG__
          std::cout<<"softmaxcrossentropy BackPropagate에서 입력에 해당하는 operator에 넘겨주는 delta : "<<'\n'<<input_delta<<'\n';
        #endif

        return NULL;
    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0);


    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0);


#endif  // __CUDNN__

    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }
};

#endif  // SOFTMAXCROSSENTROPY_H_

*/
