#ifndef __BAHDANAUDECODER2__
#define __BAHDANAUDECODER2__    value

#include "../Module.hpp"


//그래프상에서 해결하고자 함!

template<typename DTYPE> class Bahdanau2 : public Module<DTYPE>{
private:

    int timesize;     //결국 이게 MaxTimeSize랑 동일한거지!

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_encoderHidden;         // 사용 안하고 있음!!!

    Operator<DTYPE> *m_EncoderLengths;

    Operator<DTYPE> *m_Query;

public:

    Bahdanau2(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, int vocabLength, int embeddingDim, int hiddensize, int outputsize, Operator<DTYPE> *pEncoderLengths = NULL, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, pMask, pEncoderLengths, vocabLength, embeddingDim, hiddensize, outputsize, use_bias, pName);
    }


    virtual ~Bahdanau2() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, Operator<DTYPE> *pEncoderLengths, int vocabLength, int embeddingDim, int hiddensize, int outputsize, int use_bias, std::string pName) {

        this->SetInput(3, pInput, pEncoder, pMask);           //여기 Encoder도 같이 연결해줌!!!

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize), "tempHidden");

        m_EncoderLengths = pEncoderLengths;

        //중요! 핵심부분 추가!!!
        m_Query = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, hiddensize), "m_Query");

        Operator<DTYPE> *out = pInput;


        //Embedding
        Operator<DTYPE> *embedding = new EmbeddingLayer<float>(out, vocabLength, embeddingDim, pName+"_Embedding");

        Operator<DTYPE> *ContextVector = new AttentionModule<DTYPE>(pEncoder, m_Query, pEncoder, pMask, pName+"_AttentionModule");

        Operator<DTYPE> *concate = new ConcatenateColumnWise<DTYPE>(embedding,ContextVector, pName+"_concatenate");

        Operator<DTYPE> *hidden = new RecurrentLayer<DTYPE>(concate, embeddingDim+hiddensize, hiddensize, outputsize, m_initHiddenTensorholder, use_bias, pName+"_RNN");


        //linear
        out = new Linear<DTYPE>(hidden, hiddensize, outputsize, TRUE, pName+"_Fully-Connected-H2O");

        this->AnalyzeGraph(out);

        std::cout<<"연결 바꾸기 전"<<'\n';
        Container<Operator<DTYPE> *> *attention_C = ContextVector->GetInputContainer();
        std::cout<<(*attention_C)[0]->GetName()<<'\n';
        std::cout<<(*attention_C)[1]->GetName()<<'\n';
        std::cout<<(*attention_C)[2]->GetName()<<'\n';
        std::cout<<(*attention_C)[3]->GetName()<<'\n';
        //연결 바꾸기....
        ContextVector->GetInputContainer()->Pop(m_Query);
        ContextVector->GetInputContainer()->Pop(pEncoder);
        ContextVector->GetInputContainer()->Pop(pMask);
        ContextVector->GetInputContainer()->Push(hidden);
        ContextVector->GetInputContainer()->Push(pEncoder);
        ContextVector->GetInputContainer()->Push(pMask);

        std::cout<<"연결 변경 후"<<'\n';
        std::cout<<(*attention_C)[0]->GetName()<<'\n';
        std::cout<<(*attention_C)[1]->GetName()<<'\n';
        std::cout<<(*attention_C)[2]->GetName()<<'\n';
        std::cout<<(*attention_C)[3]->GetName()<<'\n';

        //현재 문제가..... AttentionModule 안에 있는 operator의 연결구조는 바뀌지않음....

        //내부의 query 연결까지 수정....
        //SetQuery까지 해주면...?
        ContextVector->SetQuery(hidden);

        return TRUE;
    }


    int ForwardPropagate(int pTime=0) {

        //std::cout<<"attention decoder forward "<<'\n';

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

        //복사하는 곳에는 문제 없음!!! 잘 해옴!!!
        // std::cout<<"Decodner에서 복사해온 encoder의 결과값"<<'\n';
        // std::cout<<initHidden->GetShape()<<'\n';
        // std::cout<<initHidden<<'\n';
        //여기까지가 복사해주는 부분!

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        // for(int ti=0; ti<timesize; ti++){
            for (int i = 0; i < numOfExcutableOperator; i++) {
                (*ExcutableOperator)[i]->ForwardPropagate(pTime);
            }
        // }

        //decoder output 확인하기!
        //std::cout<<"Bahdanau Forward 결과"<<'\n';
        // std::cout<<this->GetResult()->GetShape()<<'\n';
        // std::cout<<this->GetResult()<<'\n';

        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        //std::cout<<"----------------Decoder Backward 호출----------------"<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        // for(int ti=timesize-1; ti>=0; ti--){
            for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
                (*ExcutableOperator)[i]->BackPropagate(pTime);
                //std::cout<<(*ExcutableOperator)[i]->GetName()<<'\n';
            }
        // }

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




};


#endif  // __ATTENTIONDECODERMODULE__
