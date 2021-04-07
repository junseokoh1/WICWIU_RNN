#ifndef RECURRENTCUDNN_H_
#define RECURRENTCUDNN_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

//인터넷 방식 + 한 time에 대해서만 처리 = 방법3

template<typename DTYPE> class RecurrentCUDNN : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;



#ifdef __CUDNN__


    //Tensor descriptor

    cudnnTensorDescriptor_t x_desc[1], y_desc[1], dx_desc[1], dy_desc[1];


    cudnnTensorDescriptor_t hx_desc, dhx_desc, cx_desc, dcx_desc;

    cudnnTensorDescriptor_t hy_desc, dhy_desc, cy_desc, dcy_desc;

    //Filter descriptor
    cudnnFilterDescriptor_t w_desc, dw_desc;

    //dropout descriptor
    cudnnDropoutDescriptor_t dropout_desc;

    size_t state_size = 0;
    float m_droprate = 0.f;
    void *state = NULL;

    //RNN descriptor
    cudnnRNNDescriptor_t rnn_desc;
    cudnnRNNMode_t rnn_mode;
    cudnnRNNAlgo_t rnn_algo;


    //workspace
    void *workspace;
    void *reserved_space;

    //workspace size
    size_t workspace_size;
    size_t reserved_size;
    size_t weight_size;

    ///< cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.
    DTYPE *x;                // input
    DTYPE *hx;    // input of initial hidden state
    DTYPE *cx;    // input of cell state (LSTM)

    DTYPE *y;                // output
    DTYPE *hy;    // output of final hidden state
    DTYPE *cy;    // output of final cell state (LSTM)

    DTYPE *dy;               // input of gradient
    DTYPE *dhy;    // input of final hidden state
    DTYPE *dcy;    // input of final cell state (LSTM)

    DTYPE *dx;               // output of gradient at the input of rnn
    DTYPE *dhx;    // output of gradient at the initial hidden state
    DTYPE *dcx;    // output of gradient at the initial cell state

    DTYPE *weights;
    DTYPE *gweights;

    //RNN dimensional information
    int dimA[3];
    int strideA[3];
    int dimW[3];


    //gradient 연결해주는거 때문에 추가
    DTYPE m_alpha;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y
    DTYPE m_beta;
    ///< 연산 간 두 Operand의 가중치를 표현하기 위한 변수. ex) z = α*x + β*y



#endif  // __CUDNN__




public:
    RecurrentCUDNN(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    //pName때문에 Operator 생성자 호출이 안되는듯!!!!   숫자 4로해도 되는건가?
    RecurrentCUDNN(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    ~RecurrentCUDNN() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        AddBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");
        ApplyActivation  = new Tanh<DTYPE>(AddBias, "rnn_tanh");

        //ApplyActivation  = new Relu<DTYPE>(AddBias, "rnn_tanh");

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

#if __CUDNN__
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

        Operator<DTYPE> *pInput    = this->GetInput()[0];
        Operator<DTYPE> *pWeightIH = this->GetInput()[1];

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int colsizeOfInput = (*InputShape)[4];

        //현재 time1에 대해서만 연산을 하니깐 -> 1로 해주기
        //int hidTimeSize    = (*InputShape)[TIME];
        int hidTimeSize    = 1;
        int hidBatchSize   = (*InputShape)[BATCH];
        int hidChannelSize = (*InputShape)[2];
        int hidColSize     = (*WeightXHShape)[3];

        m_alpha = 1;
        m_beta  = 0;

        m_aTempHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

/*
        //이거 결국 hidden이랑 batch랑 channel은 같겠지?...
        int batchsizeOfWeight   = (*WeightXHShape)[1];
        int channelsizeOfWeight = (*WeightXHShape)[2];
        int rowsizeOfWeight     = (*WeightXHShape)[3];
        int colsizeOfWeight     = (*WeightXHShape)[4];
*/


//tensor descriptor
        //배열형태인 tensor descriptor


        checkCUDNN(cudnnCreateTensorDescriptor(&x_desc[0]));
        checkCUDNN(cudnnCreateTensorDescriptor(&dx_desc[0]));

        checkCUDNN(cudnnCreateTensorDescriptor(&y_desc[0]));
        checkCUDNN(cudnnCreateTensorDescriptor(&dy_desc[0]));

        //설정
        dimA[0] = hidBatchSize;
        dimA[1] = colsizeOfInput;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        checkCUDNN(cudnnSetTensorNdDescriptor(x_desc[0], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dx_desc[0], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = hidBatchSize;
        dimA[1] = hidColSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        checkCUDNN(cudnnSetTensorNdDescriptor(y_desc[0], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dy_desc[0], CUDNN_DATA_FLOAT, 3, dimA, strideA));


        dimA[0] = 1;
        dimA[1] = hidBatchSize;
        dimA[2] = hidColSize;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;



        //배열 형태가 아닌 tensor descriptor 생성
        checkCUDNN(cudnnCreateTensorDescriptor(&hx_desc));      //initial hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhx_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cx_desc));      //initial cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcx_desc));

        checkCUDNN(cudnnCreateTensorDescriptor(&hy_desc));      //fianl hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhy_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cy_desc));      //final cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcy_desc));


        //배열 형태가 아닌 tensor descriptor 설정
        //일단은 4D함수를 사용해서 하고, batch, channel, row, col로 함!!!
        checkCUDNN(cudnnSetTensorNdDescriptor(hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dhx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dcx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dhy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCUDNN(cudnnSetTensorNdDescriptor(dcy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


//dropout descriptor
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
        checkCUDNN(cudnnDropoutGetStatesSize(this->GetCudnnHandle(), &state_size));    //state size구하기
        checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, this->GetCudnnHandle(), m_droprate, state, state_size, time(NULL))); //일단은 NULL로 줌!


//rnn descriptor
        rnn_mode = CUDNN_RNN_TANH;
        rnn_algo = CUDNN_RNN_ALGO_STANDARD;

        checkCUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
        //version6로 안해도될듯? 음... 잘 모르겠다
        //numLayers : number of stacked layers = 1 로 설정해줌
        checkCUDNN(cudnnSetRNNDescriptor_v6(this->GetCudnnHandle(), rnn_desc, hidColSize, 1, dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, rnn_mode, rnn_algo, CUDNN_DATA_FLOAT));

//Bias
        checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_NO_BIAS));

//filter descriptor

        //생성
        checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
        checkCUDNN(cudnnCreateFilterDescriptor(&dw_desc));

        //설정

            //weight size 받아오기
        //237pg
        checkCUDNN(cudnnGetRNNParamsSize(this->GetCudnnHandle(), rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT));

        std::cout<<"dimW[0] : "<<weight_size / sizeof(float)<<'\n';

        dimW[0] = weight_size / sizeof(float);
        dimW[1] = 1;
        dimW[2] = 1;

        checkCUDNN(cudnnSetFilterNdDescriptor(w_desc,  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
        checkCUDNN(cudnnSetFilterNdDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));


//workspace
        checkCUDNN(cudnnGetRNNWorkspaceSize(this->GetCudnnHandle(), rnn_desc,
                                            hidTimeSize,
                                            x_desc,
                                            &workspace_size));   //seqLength 정확하지 않음, 240pg

        checkCUDNN(cudnnGetRNNTrainingReserveSize(this->GetCudnnHandle(), rnn_desc, hidTimeSize, x_desc, &reserved_size));       //seqLength 정확하지 않음, 239pg

  //      checkCudaErrors(cudaMalloc(&workspace, workspace_size));
  //      checkCudaErrors(cudaMalloc(&reserved_space, reserved_size));


        //std::cout<<"reserved_size : "<<reserved_size<<'\n';

        if (workspace_size != 0) {
            checkCudaErrors(cudaMalloc(&workspace, workspace_size));

            if (workspace == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }

        if (reserved_size != 0) {
            checkCudaErrors(cudaMalloc(&reserved_space, reserved_size));

            if (reserved_space == NULL) {
                printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                exit(-1);
            }
        }



      }

#endif  // if __CUDNN__

    //이거 해줘야되나?
    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        #if __RNNDEBUG__
        std::cout <<pTime<<"번쨰 Recurrent forward 호출" << '\n';
        #endif  // __RNNDEBUG__

        #if __RNNDBUG__
          std::cout<<"RNN의 입력값 확인"<<'\n';
          std::cout<<this->GetInput()[0]->GetResult()<<'\n';
        #endif

        m_aInput2Hidden->ForwardPropagate(pTime);

        #if __RNNDBUG__
          std::cout<<"m_aInput2Hidden forward 이후"<<'\n';
          std::cout<<m_aInput2Hidden->GetResult()<<'\n';
        #endif

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
        }
        m_aPrevActivate->ForwardPropagate(pTime);

        AddBias->ForwardPropagate(pTime);

        #if __RNNDBUG__
          std::cout<<"Tanh계산하기 전 값"<<'\n';
          std::cout<<AddBias->GetResult()<<'\n';
        #endif

        ApplyActivation->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {

      #if __RNNDEBUG__
      std::cout <<pTime<<"번쨰 Recurrent BackPropagate 호출" << '\n';
      #endif  // __RNNDEBUG__

        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
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
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }

        //std::cout<<"Gradient 앞에 time꺼 잘 갖고왔는가"<<'\n';
        //std::cout<<ApplyActivation->GetGradient()<<'\n';
        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        return TRUE;
    }


#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input    = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weightHH = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *result   = this->GetResult();

        x       = input->GetGPUData(pTime);
        weights = weightHH->GetGPUData(0);
        y       = result->GetGPUData(pTime);

        //pTime을 넣어서 GetGPUData를 호출하면 final 값을 갖고오고
        //0을 넣어서 GetGPUData를 호출하면 initial 값을 가져오고????

        Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();
        hy = tempHidden->GetGPUData(pTime);

        //여기도 timesize 1로 바꿔줘야지!
        //int timeSize        = input->GetTimeSize();
        int timeSize        = 1;

        if(pTime !=0){
          hx = tempHidden->GetGPUData(pTime-1);
        }
        else{
          hx = NULL;
        }

        //NULL해도 되는거 처리하기
        cx = NULL;
        cy = NULL;

        //api 300pg
        checkCUDNN(cudnnRNNForwardTraining(this->GetCudnnHandle(), rnn_desc, timeSize,
                                           x_desc, x,
                                           hx_desc, hx,
                                           cx_desc, cx,
                                           w_desc, weights,
                                           y_desc, y,
                                           hy_desc, hy,
                                           cy_desc, cy,
                                           workspace, workspace_size,
                                           reserved_space, reserved_size));

        return TRUE;
    }


    int BackPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input             = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta       = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weightHH          = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightHH_gradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *result            = this->GetResult();
        Tensor<DTYPE> *this_delta        = this->GetDelta();

        y           = result->GetGPUData(pTime);
        dy          = this_delta->GetGPUData(pTime);
        dhy         = this_delta->GetGPUData(pTime);

        weights     = weightHH->GetGPUData(0);

        x           = input->GetGPUData(pTime);
        dx          = input_delta->GetGPUData(pTime);

        gweights = weightHH_gradient->GetGPUData(0);

        int timeSize = 1;
        int realTimeSize        = input->GetTimeSize();

        //hx값 설정해주기
        Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();
        if(pTime !=0){
          hx = tempHidden->GetGPUData(pTime-1);
        }
        else{
          hx = NULL;
        }

        //
        Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
        dhx = tempHiddenGrad->GetGPUData(pTime);
        if (pTime != realTimeSize-1) {
            //옆에서 넘겨주는 값
            //DTYPE *dhy =

        }


        dcy = NULL;
        cx = NULL;
        dcx = NULL;

        //276pg
        checkCUDNN(cudnnRNNBackwardData(this->GetCudnnHandle(), rnn_desc, timeSize,
                                        y_desc, y,
                                        dy_desc, dy,
                                        dhy_desc, dhy,
                                        dcy_desc, dcy,
                                        w_desc, weights,
                                        hx_desc, hx,
                                        cx_desc, cx,
                                        dx_desc, dx,
                                        dhx_desc, dhx,
                                        dcx_desc, dcx,
                                        workspace, workspace_size,
                                        reserved_space, reserved_size));

        //286pg
        checkCUDNN(cudnnRNNBackwardWeights(this->GetCudnnHandle(), rnn_desc, timeSize,
                                           x_desc, x,
                                           hx_desc, hx,
                                           y_desc, y,
                                           workspace, workspace_size,
                                           dw_desc, gweights,
                                           reserved_space, reserved_size));

        return TRUE;
    }
#endif  // if __CUDNN__



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


#endif  // RECURRENTCUDNN_H_
