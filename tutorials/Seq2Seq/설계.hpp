

class my_SeqToSeq : public NeuralNetwork<float>{
private:
public:
    my_AttentionSeqToSeq(Tensorholder<float> *input1, Tensorholder<float> *input2, Tensorholder<float> *label, int vocab_length) {
        SetInput(2, input1, input2, label);

        Operator<float> *out = NULL;

        //out = new CBOW<float>(x(입력 배열), 아웃풋크기, "CBOW");
        //out = new OnehotVector<float>(x(입력 배열), 아웃풋크기, "OnehotVector");

        //embedding 추가???

        Operator<float> *mask = new PaddingAttentionMask<float>(input1, ???, 0, FALSE, "srcMasking");

        // ======================= layer 1=======================
        out = new Encoder<float>(input1, vocab_length, 32, TRUE, "Encoder");

        out = new AttentionDecoder<float>(input2, out, mask, vocab_length, 32, vocab_length, TRUE, "Decoder");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        // 1.0이 clipValue 값! 인자 하나가 더 생김!
        //현재 RMSprop clip값 = 0.5로 되어있음
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                      // Optimizer의 첫번째 인자로 parameter목록을 전달해주는거고!!!   즉 updateparameter를 할 때 넘겨주는 parameter에 대해서만 함!!!!!
        SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        //SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        //SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE

    }

    virtual ~my_SeqToSeq() {}
};





template<typename DTYPE> class AttentionDecoder : public Module<DTYPE>{
private:

    int timesize;     //결국 이게 MaxTimeSize랑 동일한거지!

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_encoderHidden;

public:

    AttentionDecoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, int inputsize, int hiddensize, int outputsize, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, inputsize, hiddensize, outputsize, use_bias, pName);
    }


    virtual ~AttentionDecoder() {}


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
        //key query value
        Operator<DTYPE> *ContextVector = new AttentionModule<DTYPE>(pEncoder, hidden, pEncoder, pMask, timesize, "attention")

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





-----------------------------------AttentionModule------------------------------------
pKey - pEncoder
pQuery - Recurrent
pValue - pEncoder

template<typename DTYPE> class AttentionModule : public Module<DTYPE>{
private:


public:

    AttentionModule(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pKey, pQuery, pValue, pMask, pName);
    }


    virtual ~AttentionModule() {}


    int Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pValue, Operator<DTYPE> *pMask,  std::string pName) {


        AttentionWeight = DotAttentionWeight(pKey, pQuery, pMask);

        //context vector를 구해주는 연산!
        // new operator2(out, pValue);                                  //5월 22일 주석처리!
        ContextVector = new AttentionByModule<DTYPE>(AttentionWeight, pValue, pName);         //5월 22일


        this->AnalyzeGraph(ContextVector);

        return TRUE;
    }
};

-------------------DotAttentionWeight--------------------------
pKey - pEncoder
pQuery - Recurrent

template<typename DTYPE> class DotAttentionWeight : public Module<DTYPE> {
private:
public:
  DotAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName = "NO NAME");
  int       Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName);
};


template<typename DTYPE> DotAttentionWeight<DTYPE>::DotAttentionWeight(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) : Module<DTYPE>(pName) {
  #ifdef __DEBUG__
  std::cout << "DotAttentionWeight<DTYPE>::DotAttentionWeight(Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, bool , std::string )" << '\n';
  #endif  // __DEBUG__

  Alloc(pKey, pQuery, pMask, pName);
}

template<typename DTYPE> int DotAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *pMask, std::string pName) {
  #ifdef __DEBUG__
  std::cout << "DotAttentionWeight<DTYPE>::Alloc(Operator<DTYPE> *pKey, Operator<DTYPE> *pQuery, Operator<DTYPE> *, bool , std::string )" << '\n';
  #endif  // __DEBUG__

  this->SetInput(3, pKey, pQuery, pMask);

  Operator<DTYPE> *out = NULL;


  // #1. weight를 구하기 위한 연산!
  out = new DotSimilarity(pKey, pQuery, "similarity")

  // #3. pMask
  //mask 방식말고... seq2seq에서 사용했던 방식으로도 가능하기는 함...
  if(pMask) {
    out = new MaskedFill<DTYPE>(out, pMask, pName+"_pMask");
  }
  // #4. softmax

  out = new Softmax<DTYPE>(out, "attention_weight");

  this->AnalyzeGraph(out);

  return TRUE;
}





--------
새로운 operator가 2개 추가되는거지!
