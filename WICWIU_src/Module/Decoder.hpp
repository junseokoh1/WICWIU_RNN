#ifndef __DECODER__
#define __DECODER__    value

#include "../Module.hpp"

/*
이거 time 안에 있고 여기서 teacherforcing 하려고 했던 버전!!!
이거는 지금 사용X
*/

template<typename DTYPE> class Decoder : public Module<DTYPE>{
private:

    int timesize;

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_EncoderLengths;

    int m_isTeacherForcing;       //이걸 추가하는게 맞나...

public:

    Decoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddensize, int outputsize, int m_isTeacherForcing = TRUE, Operator<DTYPE> *pEncoderLengths = NULL, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, vocabLength, embeddingDim, hiddensize, outputsize, m_isTeacherForcing, pEncoderLengths, use_bias, pName);
    }


    virtual ~Decoder() {}


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

        // out = new RecurrentLayer<float>(out, embeddingDim, hiddensize, outputsize, m_initHiddenTensorholder, use_bias, "Recur_1");
        out = new GRULayer<float>(out, embeddingDim, hiddensize, outputsize, m_initHiddenTensorholder, TRUE, "Recur_1");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    int ForwardPropagate(int pTime=0) {

        //Encoder의 마지막값 복사해주기!
        Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

        Shape *_initShape = _initHidden->GetShape();
        Shape *initShape = initHidden->GetShape();

        int enTimesize = _initHidden->GetTimeSize();
        int batchsize  = _initHidden->GetBatchSize();
        int colSize    = _initHidden->GetColSize();

        // for(int ba=0; ba<batchsize; ba++){
        //     for(int co=0; co<colSize; co++){
        //         (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
        //     }
        // }

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

        //복사하는 곳에는 문제 없음!!! 잘 해옴!!!
        // std::cout<<"Decodner에서 복사해온 encoder의 결과값"<<'\n';
        // std::cout<<initHidden->GetShape()<<'\n';
        // std::cout<<initHidden<<'\n';
        //여기까지가 복사해주는 부분!

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();


        if(m_isTeacherForcing){


              std::cout<<"m_isTeacherForcing Decoder Forward 호출"<<'\n';

              for(int ti=0; ti<timesize; ti++){
                  for (int i = 0; i < numOfExcutableOperator; i++) {
                      (*ExcutableOperator)[i]->ForwardPropagate(ti);
                  }
              }

              //decoder output 확인하기!
              //std::cout<<"Decoder forward 결과!"<<'\n';
              // std::cout<<this->GetResult()->GetShape()<<'\n';
              // std::cout<<this->GetResult()<<'\n';

        }
        else{     //teacher forcing 사용 X    batch도 같이 처리해줘야됨!

            std::cout<<"m_isTeacherForcing X"<<'\n';


            //decoder의 output에 접근
            Tensor<DTYPE> *pred = this->GetResult();

            //decoder의 input에 접근
            Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
            Shape *inputShape = input->GetShape();

            //Time size만큼 반복!
            for(int ti=0; ti<timesize-1; ti++){

                //한 time에 대해 forward
                for (int i = 0; i < numOfExcutableOperator; i++) {
                    (*ExcutableOperator)[i]->ForwardPropagate(ti);
                }

                //batch만큼 반복!
                for(int ba = 0; ba < batchsize; ba++){
                    int pred_index = DecoderGetMaxIndex(pred, ba, ti, pred->GetColSize());

                    std::cout<<pred_index<<" ";

                    //다음 time의 input 설정!
                    (*input)[Index5D(inputShape, ti+1, ba, 0, 0, 0)] = pred_index;
                }
                // std::cout<<this->GetInput()[0]->GetName()<<'\n';
            }

            //마지막 time에는 그냥 forward만 하면되니깐!
            for (int i = 0; i < numOfExcutableOperator; i++) {
                (*ExcutableOperator)[i]->ForwardPropagate(timesize-1);
            }


        }


        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        // std::cout<<"----------------Decoder Backward 호출----------------"<<'\n';

        // std::cout<<"decoder가 계산한 init_hidden의 gradient값"<<'\n';
        // std::cout<<m_initHiddenTensorholder->GetGradient()<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=timesize-1; ti>=0; ti--){
            for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
                (*ExcutableOperator)[i]->BackPropagate(ti);
                //std::cout<<(*ExcutableOperator)[i]->GetName()<<'\n';
            }
        }


        // std::cout<<"Decoder last operator가 갖고 있는 gradient 값"<<'\n';
        // std::cout<<this->GetGradient()->GetShape()<<'\n';
        // std::cout<<this->GetGradient()<<'\n';

/*
        template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetGradient() const {
            return m_pLastOperator->GetGradient();
        }
        이런 함수가 존재해서 이제 GetGradient로 접근가능!!!
*/
        //Encoder로 넘겨주기!!!
        //encoder의 마지막 time에만 넘겨주면됨!!!
        Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

        Shape *enShape  = enGradient->GetShape();
        Shape *_enShape = _enGradient->GetShape();

        int enTimesize = enGradient->GetTimeSize();
        int batchSize = enGradient->GetBatchSize();
        int colSize = enGradient->GetColSize();

        // std::cout<<"decoder가 계산한 init_hidden의 gradient값"<<'\n';
        // std::cout<<_enGradient<<'\n';

        //encoder의 Gradient에 저장해주기!!!  굳이 hidden에 저장해주지 않아도 되는게 hidden2ouptut은 rnn에 없어서! 옆에서 주나 위에서 주나 똑같음!!
        //근데 결국 이게 hidden에 저장하는거 맞지않나?...
        //+=으로 수정해야 되는거 아닌가!!!  중요!!!!!!!!!!!!!!!!!!!!!!!!!중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요중요
        //encoder의 last operator는 RNN임
        // for(int ba=0; ba < batchSize; ba++){
        //     for(int co=0; co < colSize; co++){
        //         (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
        //     }
        // }

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


        return TRUE;
    }

    int DecoderGetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass) {
        Shape *pShape = data->GetShape();
        int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
        int    end    = start + numOfClass;

        // Initial max value is first element
        DTYPE max       = (*data)[start];
        int   max_index = 0;

        for (int dim = start + 1; dim < end; dim++) {
            if ((*data)[dim] > max) {
                max       = (*data)[dim];
                max_index = dim - start;
            }
        }

        return max_index;
    }


};


#endif  // __DECODER__
