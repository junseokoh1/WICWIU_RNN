#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"


class my_AttentionSeqToSeq : public NeuralNetwork<float>{
private:
public:
    my_AttentionSeqToSeq(Tensorholder<float> *EncoderInput, Tensorholder<float> *DecoderInput, Tensorholder<float> *label, Tensorholder<float> *EncoderLengths, Tensorholder<float> *DecoderLengths, int vocabLength, int embeddingDim, int hiddenDim) {
        SetInput(5, EncoderInput, DecoderInput, label, EncoderLengths, DecoderLengths);

        Operator<float> *out = NULL;

        //out = new CBOW<float>(x(입력 배열), 아웃풋크기, "CBOW");
        //out = new OnehotVector<float>(x(입력 배열), 아웃풋크기, "OnehotVector");

        //중요!!! 여기에 operator를 추가하면 time이 안돌아감!....
        //그리고 embedding할때도.... EncoderInput하고 DecoderInput하고 dim이 다름....
        //embedding - 그래서 embedding은 encoder하고 decoder 내부로 들어가야 됨!!!
        //  -
        //  - pytorch에서도 그렇게 길이 다르게 해서 for문 돌림!

        Operator<float> *mask = new PaddingAttentionMaskRNN<float>(EncoderInput, 0, "srcMasking");        //음.... 이거때문에 흠....

        // ======================= layer 1=======================
        out = new Encoder<float>(EncoderInput, vocabLength, embeddingDim, hiddenDim, TRUE, "Encoder");

        // out = new AttentionDecoder_Module<float>(DecoderInput, out, mask, vocabLength, embeddingDim, hiddenDim, vocabLength, EncoderLengths, TRUE, "Decoder");

        out = new Bahdanau2<float>(DecoderInput, out, mask, vocabLength, embeddingDim, hiddenDim, vocabLength, EncoderLengths, TRUE, "Decoder");

        // ContextVector = new AttentionModule<float>(enc, dec, mask, "attention");
        //
        // concate = new ConcatenateColumnWise<float>(dec, ContextVector);
        //
        // out = new Linear<float>(out, 1024, 10, TRUE, "Fully-connected_2");

        AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        // SetLossFunction(new HingeLoss<float>(out, label, "HL"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));
        //SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));
        SetLossFunction(new SoftmaxCrossEntropy_padding<float>(out, label, DecoderLengths, "SCE"));
        // SetLossFunction(new CrossEntropy<float>(out, label, "CE"));

        // ======================= Select Optimizer ===================
        // 1.0이 clipValue 값! 인자 하나가 더 생김!
        //현재 RMSprop clip값 = 0.5로 되어있음
        //SetOptimizer(new GradientDescentOptimizer<float>(GetParameter(), 0.001, 0.9, 1.0, MINIMIZE));                      // Optimizer의 첫번째 인자로 parameter목록을 전달해주는거고!!!   즉 updateparameter를 할 때 넘겨주는 parameter에 대해서만 함!!!!!
        // SetOptimizer(new RMSPropOptimizer<float>(GetParameter(), 0.01, 0.9, 1e-08, FALSE, MINIMIZE));
        // SetOptimizer(new AdamOptimizer<float>(GetParameter(), 0.001, 0.9, 0.999, 1e-08, MINIMIZE));
        // SetOptimizer(new NagOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));
        SetOptimizer(new AdagradOptimizer<float>(GetParameter(), 0.001, 0.9, MINIMIZE));      //MAXIMIZE

    }

    virtual ~my_AttentionSeqToSeq() {}

    //이렇게 사용하려고 하면 NN에서 virtual로 선언해주면됨!
    int seq2seqBPTT(int EncTimeSize, int DecTimeSize);
    int seq2seqBPTTOnGPU(int EncTimeSize, int DecTimeSize);
    int SentenceTranslate(std::map<int, std::string>* index2vocab);
};

int my_AttentionSeqToSeq::seq2seqBPTT(int EncTimeSize, int DecTimeSize) {

    // std::cout<<"my_AttentionSeqToSeq에 있는 함수"<<'\n';

    LossFunction<float> *m_aLossFunction = this->GetLossFunction();
    Optimizer<float> *m_aOptimizer = this->GetOptimizer();

    this->ResetResult();
    this->ResetGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    Container<Operator<float> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

    //maks forward
    (*ExcutableOperator)[0]->ForwardPropagate(0);

    //encoder forward
    for(int i =0; i<EncTimeSize; i++)
        (*ExcutableOperator)[1]->ForwardPropagate(i);

    // std::cout<<"seq2seqBPTT Encoder forward 완료"<<'\n';

    //Decoder & lossfunction forward
    for(int i=0; i<DecTimeSize; i++){
      // std::cout<<"-------------"<<i<<"----------------";
      (*ExcutableOperator)[2]->ForwardPropagate(i);
      m_aLossFunction->ForwardPropagate(i);
    }

    // std::cout<<"seq2seqBPTT forward 완료"<<'\n';

    //Decoder & loss function backward
    for(int j=DecTimeSize-1; j>=0; j--){
      m_aLossFunction->BackPropagate(j);
      (*ExcutableOperator)[2]->BackPropagate(j);
    }

    //Encoder backward
    for(int j=EncTimeSize-1; j>=0; j--){
      (*ExcutableOperator)[1]->BackPropagate(j);
    }

    m_aOptimizer->UpdateParameter();

    return TRUE;
}

int my_AttentionSeqToSeq::seq2seqBPTTOnGPU(int EncTimeSize, int DecTimeSize) {
#ifdef __CUDNN__
    this->ResetResult();
    this->ResetGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    LossFunction<float> *m_aLossFunction = this->GetLossFunction();
    Optimizer<float> *m_aOptimizer = this->GetOptimizer();

    Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

    //encoder forward
    for(int i =0; i<EncTimeSize; i++)
        (*ExcutableOperator)[0]->ForwardPropagateOnGPU(i);

    //Decoder & lossfunction forward
    for(int i=0; i<DecTimeSize; i++){
      (*ExcutableOperator)[1]->ForwardPropagateOnGPU(i);
      m_aLossFunction->ForwardPropagateOnGPU(i);
    }

    //Decoder & loss function backward
    for(int j=DecTimeSize-1; j>=0; j--){
      m_aLossFunction->BackPropagateOnGPU(j);
      (*ExcutableOperator)[1]->BackPropagateOnGPU(j);
    }

    //Encoder backward
    for(int j=EncTimeSize-1; j>=0; j--){
      (*ExcutableOperator)[0]->BackPropagateOnGPU(j);
    }

    m_aOptimizer->UpdateParameterOnGPU();
#else
    std::cout<<"There is no GPU option!"<<'\n';
    exit(-1);
#endif

    return TRUE;
}

int my_AttentionSeqToSeq::SentenceTranslate(std::map<int, std::string>* index2vocab){

    // std::cout<<"my_AttentionSeqToSeq내에 있는 SentenceTranslate 호출"<<'\n';

    //Result
    Tensor<float> *pred = this->GetResult();

    //DecoderInput
    Tensor<float> *DecoderInput = this->GetInput()[1]->GetResult();
    Shape *InputShape = DecoderInput->GetShape();

    //encoder, decoder time size
    int EncoderTimeSize = this->GetInput()[0]->GetResult()->GetTimeSize();
    int DecoderTimeSize = DecoderInput->GetTimeSize();

    //Encoder, Decoder module access
    int numOfExcutableOperator = this->GetNumOfExcutableOperator();
    Container<Operator<float> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

    // std::cout<<"num : "<<numOfExcutableOperator<<'\n';
    // std::cout<<(*ExcutableOperator)[0]->GetName()<<'\n';
    // std::cout<<(*ExcutableOperator)[1]->GetName()<<'\n';

    //maks forward
    (*ExcutableOperator)[0]->ForwardPropagate(0);

    //encoder forward
    for(int ti = 0; ti < EncoderTimeSize; ti++)
        (*ExcutableOperator)[1]->ForwardPropagate(ti);

    //decoder input holder tensor에 접근해서?....
    // shape : [DecoderTime, BATCH, 1, 1, 1]

    //첫번째 입력은 SOS
    (*DecoderInput)[0] = 1;

    for(int ti = 0; ti < DecoderTimeSize; ti++){

        //decoder forward
        (*ExcutableOperator)[2]->ForwardPropagate(ti);

        int pred_index = this->GetMaxIndex(pred, 0, ti, pred->GetColSize());

        //출력하기!
        std::cout<<pred_index<<" : ";
        std::cout<<index2vocab->at(pred_index)<<'\n';

        //EOS이면 끝내기!
        if( pred_index == 2)
          break;

        //결과값 입력값으로 복사!
        if(ti != DecoderTimeSize-1){
            (*DecoderInput)[Index5D(InputShape, ti+1, 0, 0, 0, 0)] = pred_index;
        }
    }

    //test 완료후 reset
    this->ResetResult();

    //(*decoder_x_holder0[Index5D(DecoderInputShape, 0, 0, 0, 0, 0)])

    // std::cout<<(*ExcutableOperator)[1]->GetNumOfExcutableOperator()<<'\n';

    //std::cout<<"SentenceTranslate 완료"<<'\n';

}
