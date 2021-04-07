#ifndef __DECODER__
#define __DECODER__    value

#include "../Module.hpp"


template<typename DTYPE> class Decoder : public Module<DTYPE>{
private:

    int timesize;     //결국 이게 MaxTimeSize랑 동일한거지!

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_encoderHidden;

public:

    Decoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, inputsize, hiddensize, outputsize, use_bias, pName);
    }


    virtual ~Decoder() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int inputsize, int hiddensize, int outputsize, int use_bias, std::string pName) {

        this->SetInput(2, pInput, pEncoder);           //여기 Encoder도 같이 연결해줌!!!

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize), "tempHidden");

        Operator<DTYPE> *out = pInput;

        //pEncoder        ????

        //------------------------------weight 생성-------------------------
        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);

        Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize*2, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);

        Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        //여기에 attention을 추가하기???
        //이런식으로 처리하면 될 듯???
        //out = new attention(out, ...)

        Operator<DTYPE> *hidden = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias, m_initHiddenTensorholder);                           //tensor 넘겨주는지 operator 넘겨주는지 이걸로ㄱㄱ!!!

        //값 어떻게 줄껀데...
        Operator<DTYPE> *ContextVector = new DotAttention<DTYPE>(pEncoder, pEncoder, timesize, "attention")         // 이걸 또 수정해서 softmax 전에 MaskedFill operator를 호출하도록 해야함!!!!

        //attention함수의 setQuery함 한번만 호출하면 됨 !
        ContextVector->SetQuery(hidden);

        //out = new MaskedFill<DTYPE>(ContextVector, pMask, pName+"_pMask");                                    //이거는 softmax 전에 해야되는 거니깐!!! DotAttention 내부에 들어가야 되는거임!!!

        out = new ConcatenateColumnWise(hidden,ContextVector, "concatenate");

        out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        //tanh이 들어가야 하는가??? - 이거는 논문에서 찾아보자!


        this->AnalyzeGraph(out);

        return TRUE;
    }

    int ForwardPropagate(int pTime=0) {

        //std::cout<<"Decoder Forward 호출"<<'\n';

        //Encoder의 마지막값 복사해주기!
        Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

        Shape *_initShape = _initHidden->GetShape();
        Shape *initShape = initHidden->GetShape();

        int enTimesize = _initHidden->GetTimeSize();
        int batchsize  = _initHidden->GetBatchSize();
        int colSize    = _initHidden->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int co=0; co<colSize; co++){
                (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];     //padding을 추가한다면 이 부분이 수정이 필요! ba의 값에따라 enTimesize가 바뀌어야 함!!!
            }
        }

        //복사하는 곳에는 문제 없음!!! 잘 해옴!!!
        // std::cout<<"Decodner에서 복사해온 encoder의 결과값"<<'\n';
        // std::cout<<initHidden->GetShape()<<'\n';
        // std::cout<<initHidden<<'\n';
        //여기까지가 복사해주는 부분!

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=0; ti<timesize; ti++){
            for (int i = 0; i < numOfExcutableOperator; i++) {
                (*ExcutableOperator)[i]->ForwardPropagate(ti);
            }
        }

        //decoder output 확인하기!
        // std::cout<<"Decoder forward 결과!"<<'\n';
        // std::cout<<this->GetResult()->GetShape()<<'\n';
        // std::cout<<this->GetResult()<<'\n';

        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        //std::cout<<"----------------Decoder Backward 호출----------------"<<'\n';

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for(int ti=timesize-1; ti>=0; ti--){
            for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
                (*ExcutableOperator)[i]->BackPropagate(ti);
                //std::cout<<(*ExcutableOperator)[i]->GetName()<<'\n';
            }
        }

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
        for(int ba=0; ba < batchSize; ba++){
            for(int co=0; co < colSize; co++){
                (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];   //+=으로 수정해야 되는거 아닌가!!!
            }
        }


        return TRUE;
    }

};


#endif  // __DECODER__
