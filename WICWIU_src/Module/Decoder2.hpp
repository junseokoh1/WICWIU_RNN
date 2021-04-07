#ifndef __DECODER2__
#define __DECODER2__    value

#include "../Module.hpp"


template<typename DTYPE> class Decoder2 : public Module<DTYPE>{
private:

    int timesize;

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_EncoderLengths;

    int m_isTeacherForcing;       //이걸 추가하는게 맞나...      이거는 나중에 삭제하자!!!

public:

    Decoder2(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddensize, int outputsize, int m_isTeacherForcing = TRUE, Operator<DTYPE> *pEncoderLengths = NULL, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, vocabLength, embeddingDim, hiddensize, outputsize, m_isTeacherForcing, pEncoderLengths, use_bias, pName);
    }


    virtual ~Decoder2() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddensize, int outputsize, int teacherForcing, Operator<DTYPE> *pEncoderLengths, int use_bias, std::string pName) {

        this->SetInput(2, pInput, pEncoder);           //여기 Encoder도 같이 연결해줌!!!

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize), "tempHidden");

        //padding때문에 추가!
        m_EncoderLengths = pEncoderLengths;
        m_isTeacherForcing = teacherForcing;
        //teacheringforcing 때문에 추가


        Operator<DTYPE> *out = pInput;

        //pEncoder        ????

        //------------------------------weight 생성-------------------------
        // Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, embeddingDim, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        // //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        // Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);
        //
        // Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);
        //
        // Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        //Embedding 추가!!!
        out = new EmbeddingLayer<float>(out, vocabLength, embeddingDim, "Embedding");

        // out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias, m_initHiddenTensorholder);                           //tensor 넘겨주는지 operator 넘겨주는지 이걸로ㄱㄱ!!!
        //
        // out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");
        //
        // if (use_bias) {
        //     Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
        //     out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        // }

          out = new RecurrentLayer<float>(out, embeddingDim, hiddensize, outputsize, m_initHiddenTensorholder, use_bias, "Recur_1");
        //out = new LSTM2Layer<float>(out, embeddingDim, hiddensize, outputsize, m_initHiddenTensorholder, TRUE, "Recur_1");
        //out = new GRULayer<float>(out, embeddingDim, hiddensize, m_initHiddenTensorholder, TRUE, "Recur_1");


        //이제 h2o을 밖으로!
        out = new Linear<float>(out, hiddensize, outputsize, TRUE, "Fully-Connected-H2O");

        this->AnalyzeGraph(out);

        return TRUE;
    }


/*
    //************************************************************************************gethidden사용 하기 전에 FowardBackward****************************************************************************
    int ForwardPropagate(int pTime=0) {

        //std::cout<<"Decoder forward "<<pTime<<'\n';

        if(pTime == 0){
              //Encoder의 마지막값 복사해주기!
              Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
              Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

              Shape *_initShape = _initHidden->GetShape();
              Shape *initShape = initHidden->GetShape();

              int enTimesize = _initHidden->GetTimeSize();
              int batchsize  = _initHidden->GetBatchSize();
              int colSize    = _initHidden->GetColSize();

              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
                      }
                  }
              }
              else{
                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
                      }
                  }
              }
        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++)
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);

        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        // std::cout<<"----------------Decoder Backward 호출----------------"<<'\n';


        // if(pTime == timesize-1){
        //     std::cout<<"initHidden gradient"<<'\n';
        //     std::cout<<m_initHiddenTensorholder->GetGradient()<<'\n';
        // }


        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
            //std::cout<<(*ExcutableOperator)[i]->GetName()<<'\n';
        }


        if(pTime == 0){

              //std::cout<<m_initHiddenTensorholder->GetGradient()<<'\n';

              //Encoder로 넘겨주기!!!
              //encoder의 마지막 time에만 넘겨주면됨!!!
              Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
              Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

              Shape *enShape  = enGradient->GetShape();
              Shape *_enShape = _enGradient->GetShape();

              int enTimesize = enGradient->GetTimeSize();
              int batchSize = enGradient->GetBatchSize();
              int colSize = enGradient->GetColSize();


              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
                      }
                  }

              }
              else{
                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
                      }
                  }
              }

        }

        return TRUE;
    }
    //************************************************************************************gethidden사용 하기 전에 FowardBackward**************************************************************
*/



    int ForwardPropagate(int pTime=0) {


        if(pTime == 0){
              //Encoder의 마지막값 복사해주기!
              Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
              Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

              Shape *_initShape = _initHidden->GetShape();
              Shape *initShape = initHidden->GetShape();

              int enTimesize = _initHidden->GetTimeSize();
              int batchsize  = _initHidden->GetBatchSize();
              int colSize    = _initHidden->GetColSize();

              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
                      }
                  }
              }
              else{
                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
                      }
                  }
              }
        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++)
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);

        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        // std::cout<<"----------------Decoder Backward 호출----------------"<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
            //std::cout<<(*ExcutableOperator)[i]->GetName()<<'\n';
        }


        if(pTime == 0){

              Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
              Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

              Shape *enShape  = enGradient->GetShape();
              Shape *_enShape = _enGradient->GetShape();

              int enTimesize = enGradient->GetTimeSize();
              int batchSize = enGradient->GetBatchSize();
              int colSize = enGradient->GetColSize();


              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
                      }
                  }

              }
              else{
                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
                      }
                  }
              }

        }

        return TRUE;
    }


    #if __CUDNN__

    template<typename DTYPE> int Module<DTYPE>::ForwardPropagateOnGPU(int pTime) {


        //encoder에서 decoder로 복사!
        if(pTime == 0){



        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < m_numOfExcutableOperator; i++) {
            (*m_aaExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
        }
        return TRUE;
    }

    /*!
     * @brief GPU를 이용해 모듈 그래프의 역전파를 수행하는 메소드
     * @details 역순으로 Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::BackPropagateOnGPU(int pTime) 메소드를 호출한다.
     * @param pTime 각 BackPropagateOnGPU 메소드에 전달할 Time의 인덱스
     * @return TRUE
     */
    template<typename DTYPE> int Module<DTYPE>::BackPropagateOnGPU(int pTime) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
            (*m_aaExcutableOperator)[i]->BackPropagateOnGPU(pTime);
        }
        return TRUE;
    }


    #endif // CUDNN

};


#endif  // __DECODER__
