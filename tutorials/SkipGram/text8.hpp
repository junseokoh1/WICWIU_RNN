#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include <time.h>   //난수 생성위해 추가!

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"


//여기에 선언해주는 숫자는!!! 이제 파일에 있는 단어의 개수임!!!
//text8꺼 다시 확인해보고 사용하기!
//#define NUMOFVOCAB      71293        //text8에서 중복 없는 단어의 개수!(vocab size)  //  eos sos +2 해주면 71293개!   //eos만 해주면 +1 71292

//이걸 사용!
//#define NUMOFVOCAB      253856        //text8에서 중복 없는 단어의 개수!(vocab size)  //  eos sos +2 해주면 71293개!   //eos만 해주면 +1 71292

//#define NUMOFVOCAB      4181        //shakespear.txt 에서 중복 없는 단어의 개수!(vocab size)  // eos, sos 추가 안하면 4179 //  eos sos +2 해주면 4181개!
//매우중요!!! 단어개수가!!!.... 이게 결국 0부터 시작하면 4180개 이고.... 1부터 시작하면 4181개인거임....
//#define NUMOFVOCAB      2515            //Subtext8.txt
#define NUMOFVOCAB        15036            //Subtext8-2.txt
//#define NUMOFVOCAB      1289               //debug.txt


//#define NUMOFWORD       17005207     //text8에서 단어의 개수!
//#define NUMOFWORD       27655        //shakespeare.txt에서 단어의 개수!
//#define NUMOFWORD       9967          //subtext8.txt에서 단어의 개수!
#define NUMOFWORD         142395          //subtext8-2.txt에서 단어의 개수!
//#define NUMOFWORD         4278           //debug.txt


using namespace std;

enum OPTION {
    ONEHOT,
    CBOWMODE,
    SKIPGRAM,
    ACCURACY
};


void MakeOneHotVector(int* onehotvector, int vocab_size, int index){

    for(int i=0; i<vocab_size; i++){
        if(i==index)
            onehotvector[i] = 1;
        else
            onehotvector[i] = 0;
    }
}


void Eliminate(char *str, char ch){

    int length = strlen(str);
    for(int i=0; i<length; i++){

        if(str[i] == ch)
        {
            for(int j=i; j<length; j++)
                str[j] = str[j+1];            //+1로 처리해주기 때문에 NULL까지 옮겨줌!!!
        }
    }
}

void replace(char *str, char tar, char repl){

    int length = strlen(str);
    for(int i=0; i<length; i++){
        if(str[i] == tar)
           str[i] = repl;
    }
}


template<typename DTYPE>
class text8 : public Dataset<DTYPE> {
private:

    //textData에 있던 변수들!!!
    string* vocab;          //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;         //파일에서 읽어오기!

    string* wordTextData;   //strtok를 사용하면 원래 data가 바뀌어서 추가한거!

    int vocab_size;         //반복없는 단어의 개수
    int text_length;        // 이거는 char 개수...     //나중에 fastText에서 필요할 수도 있을거 같아서 남겨둠!!!
    int word_num;           //단어의 개수

    //각 단어가 몇 번 나왔는지는 없음...! 이거 배열을 하나 더 만들어서 가능할듯!!!   -> sampling 할때 필요!
    int* wordFrequency;     //이걸.... 음.... vocab size를 미리 받아서 하는걸로 할까.... 아니면... 음... 우찌하면 좋을라나...


    OPTION option;

    //word2vec.hpp에 있던 거!!!
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;             //input data의 개수!!!
    int m_window;                 //window size -> 홀수가 기본이겠지!
    int m_negative;

    int m_dimOfInput;
    int m_dimOfLabel;


public:
    text8(string File_Path, int window, int negative, OPTION pOption) {
          vocab = NULL;
          TextData = NULL;
          wordTextData = NULL;

          vocab_size = 0;
          text_length = 0;
          word_num=0;

          option = pOption;

          //word2vec.hpp에 있던거!
          m_aaInput = NULL;
          m_aaLabel = NULL;

          m_numOfInput = 0;
          m_window     = 0;
          m_negative   = 0;

          m_dimOfInput = 0;
          m_dimOfLabel = 0;

          Alloc(File_Path, window, negative);
    }

    virtual ~text8() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string pTextPath, int window, int negative);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM
    void                                  MakeSkipGramInputData();
    void                                  MakeSkipGramLabelData();
    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM

    //ACCURACYACCURACYACCURACYACCURACYACCURACYACCURACYACCURACY
    void                                  MakeAccuracyData();
    //ACCURACYACCURACYACCURACYACCURACYACCURACYACCURACYACCURACY

    int                                   word2index(string str);

    string                                index2word(int index);

    int                                   GetTextLength();

    int                                   GetWordNum();

    string*                               GetVocab();

    int                                   GetVocabSize();

    int                                   GetInputDim();
    int                                   GetLabelDim();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void text8<DTYPE>::Alloc(string File_Path, int window, int negative) {


    vocab        = new string[NUMOFVOCAB];      //원래는 NUMOFWORD+2 였음!  이거 문제가 아님!!!
    wordTextData = new string[NUMOFWORD];

    //subsampling 때문에 추가됨!
    wordFrequency = new int[NUMOFVOCAB];

    m_window     =  window;
    m_negative   = negative;

    if(option == SKIPGRAM){

      std::cout<<"skipgram alloc 호출!"<<'\n';

      m_numOfInput = NUMOFWORD * (m_window - 1);     //sos eos 추가
      m_dimOfInput = m_negative + 2;                 //center word와 context neighbor word 2개 추가!
      m_dimOfLabel = m_negative + 1;                 //positive sample 추가
    }
    else if(option == ACCURACY){

      std::cout<<"accuracy alloc 호출!"<<'\n';

      m_numOfInput = NUMOFWORD / 4;
      m_dimOfInput = 3;
      m_dimOfLabel = 1;
    }

    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfInput];


    FileReader(File_Path);


    //여기서부터-----------------------------------------------
    //make_vocab
    MakeVocab();

    if(option == SKIPGRAM){
      MakeSkipGramInputData();
      // MakeSkipGramLabelData();
    }
    else if(option == ACCURACY){
      MakeAccuracyData();
    }
    //여기까지 alloc함수에서 호출하지 말고 다른곳에서 호출하도록 만들자!!!

}


template<typename DTYPE> void text8<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void text8<DTYPE>::FileReader(string pFile_Path) {

    std::cout<<"FileReader"<<'\n';

    ifstream fin;
    fin.open(pFile_Path);

    if(fin.is_open()){

      //파일 사이즈 구하기
      fin.seekg(0, ios::end);
      text_length = fin.tellg();
      fin.seekg(0, ios::beg);        //포인터를 다시 시작위치로 바꿈

      //파일 길이만큼 할당
      TextData = new char[text_length];

      //파일 읽기
      fin.read(TextData, text_length);

      //*************************************************************************이거 text8에 대해서는 필요 없어서 실행 시키지 않기!!!
      //소문자로 변환
      // for(int i=0; i<text_length; i++){
      //     TextData[i] = tolower(TextData[i]);
      // }

      // std::cout<<"제거 전 text 길이 : "<<text_length<<'\n';
      // std::cout<< strlen(TextData)<<'\n';

      //빈공간 다 없애고 sp만 남기기
      //*************************************************************************이거 text8에 대해서는 필요 없어서 실행 시키지 않기!!!
      // Eliminate(TextData, '\n');
      // Eliminate(TextData, ':');
      // Eliminate(TextData, ',');
      // Eliminate(TextData, '.');
      // Eliminate(TextData, ';');
      // Eliminate(TextData, '?');
      // Eliminate(TextData, '!');
      // replace(TextData, '\r', ' ');

    }

    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    fin.close();
}



template<typename DTYPE> void text8<DTYPE>::MakeVocab(){

    std::cout<<"---------------------Makevocab------------------"<<'\n';

    int flag = 0;
    char* token = strtok(TextData, " ");    //단어 하나씩 가져오기 위한 변수

    int word_count =0;

    while(token != NULL){

          //wordTextData[NUMOFWORD-4] = token;
          wordTextData[word_count] = token;         //301291

           //std::cout<<"word count : "<<word_count<<" "<<token<<'\n';
           //std::cout<<word_count<<'\n';

           if(word_count%100000==0)
               std::cout<<word_count<<'\n';

          //중복확인하기
          for(int i=0; i<vocab_size; i++){
              if(vocab[i] == token){
                  flag = 1;
                  wordFrequency[i]++;
                  break;
                  // std::cout<<"중복된 단어 : "<<token<<'\n';
              }
          }

          //중복이 아니라면
          if(flag == 0){
              vocab[vocab_size] = token;
              wordFrequency[vocab_size] = 1;
              vocab_size++;
          }

          token = strtok(NULL, " ");
          //단어 개수
          word_count++;
          flag = 0;
    }

    //SOS하고 EOS 추가하려면 여기서!!! 그리고 vocab수하고 파일에 있는 단어수도 증가시키기???
    //sos : "<s>"    eos : "<\s>"

    // std::cout<<index2word(vocab_size-1)<<" "<<index2word(vocab_size)<<"???"<<'\n';

    wordFrequency[vocab_size] = 1;
    vocab[vocab_size++] = "<s>";
    wordFrequency[vocab_size] = 1;
    //매우 중요!!! 여기서 ++해주는 이유는!!! 이제 모든 index 접근을.... <=가 아닌 <로 for문을 만들어서... 문제가 생김!! 이 문제는 단순히 text8에서만의 문제가 아니라!!! 다른 operator에서도 이제 접근할 때 for문에서 <로 해서 이제 문제가 생김!!! 마지막 <e>이거 때문에!!!
    vocab[vocab_size++] = "<e>";

    //count를 사용해서 subsampling을 하기위해!!! 굳이 sort를 할 필요가 없다고 생각함!!! 결과에 영향 없을 듯!
    //sort(vocab, vocab+vocab_size-1);

    word_num = word_count;

    std::cout<<"파일에 있는 단어의 개수(중복 포함) : "<<word_num<<'\n';
    std::cout<<"파일에 있는 단어의 개수(중복 미포함) : "<<vocab_size<<'\n';
    // std::cout<<index2word(vocab_size-2)<<" : "<<wordFrequency[vocab_size-2]<<" "<<index2word(vocab_size-1)<<" : "<<wordFrequency[vocab_size-1]<<'\n';
    // std::cout<<word2index("<s>")<<" "<<word2index("<e>")<<'\n';
    // std::cout<<"work : "<<wordFrequency[word2index("work")]<<'\n';

}


//일단 subsampling of frequent words - 이거는 일단 제외하고 하겠음!!
// center  context  non-context  non-context
template<typename DTYPE> void text8<DTYPE>::MakeSkipGramInputData(){

      //subsampling때문에 추가
      unsigned long long next_random = 1;       // 출력은 %lld임!
      float sample = 1e-4;
      //여기까지 subsamling때문에 추가


      //이게 시간이 오래걸리지 않을까?....
      std::cout<<"------------------------------MakeskipgramInputdata---------------------"<<'\n';

      srand(time(NULL));                      //이 함수는 main에서 한번만 호출하면 됨!!!!   이거 그래서 수정 필요!!!
      int offsetIndex = 0;
      int index = 0;

      //context에 해당하는 index에 접근하기 위한 offset
      int contextOffset[m_window-1];
      for(int i=0; i<m_window/2; i++){
          contextOffset[index++] = i+1;
          contextOffset[index++] = -(i+1);
      }

      int centerIndex     = 0;
      int nonContextIndex = 0;

      for (int i = 0; i < m_numOfInput; i++) {

            //min count에 못 미치면 그냥 pass!!!
            //wordFrequency[word2index(wordTextData[nonContextIndex]

            m_aaInput[i] = new DTYPE[m_dimOfInput];

            //center word
            m_aaInput[i][0] = (DTYPE)word2index(wordTextData[centerIndex]);


            //context word + sos + eos = positive sample
            if(centerIndex+contextOffset[offsetIndex] < 0){
                m_aaInput[i][1] = (DTYPE)word2index("<s>");
            } else if(centerIndex+contextOffset[offsetIndex] >= word_num){
                m_aaInput[i][1] = (DTYPE)word2index("<e>");
            } else{
                m_aaInput[i][1] = (DTYPE)word2index(wordTextData[ centerIndex+contextOffset[offsetIndex] ]);
            }

            //non-context word = negative sample
            //일단은  negative sample에만 subsampling 적용하기! -> pytorch는 그렇게함!!
            for (int d = 2; d < m_dimOfInput; d++) {
                  nonContextIndex = rand()%word_num;


                  //sub sampling 추가해보기!!!
                  //next_random값 !!!! 중요!!! 이거 dataset에 따라 값을 바꿔줄 필요가 있음!!! data가 작아지면 값이 전체적으로 작아져서 무한loop에 들어감!!!
                  //이거 근데 로직이 이상한게..... 한번 안되면 계속 안되는거 아닌가....???
                  // next_random = 1;
                  // float ran = (sqrt(wordFrequency[word2index(wordTextData[nonContextIndex])] / (sample * word_num)) + 1) * (sample * word_num) / wordFrequency[word2index(wordTextData[nonContextIndex])];                                    //여기서 ran설정...
                  // next_random = (unsigned long long)25214903917 + 11;
                  // //std::cout<<wordFrequency[word2index(wordTextData[nonContextIndex])]<<"          "<<ran<<" "<<(next_random & 0xFFFF) / (float)65536<<'\n';
                  // if (ran < (next_random & 0xFFFF) / (float)65536 ){                                                                                              //ran은 여기서만 사용....
                  //     d--;
                  //     continue;
                  // }


                  //center와 겹치지 않도록!!!
                  if(nonContextIndex == centerIndex){
                       d--;
                       // std::cout<<"center와 겹침!"<<'\n';
                       continue;
                  }

                  //context에 대한 처리 추가!
                  m_aaInput[i][d] = (DTYPE)word2index(wordTextData[nonContextIndex]);
                  for(int j=0; j<m_window-1; j++){
                      if(nonContextIndex==centerIndex+contextOffset[j]){
                          // std::cout<<"centext와 겹침!"<<'\n';
                          d--;
                      }
                  }
            }
            // std::cout<<'\n'<<'\n';

            offsetIndex++;
            if(offsetIndex == m_window-1){
                centerIndex++;
                offsetIndex = 0;
            }

            //이제 입력 확인해보기!
            // for(int j=0; j<m_dimOfInput; j++)
            //     std::cout<<index2word(m_aaInput[i][j])<<" ";
            // std::cout<<'\n';
      }
}

//이 함수 굳이 필요는 없음....   필요 없을듯??? 일단 수정한 곳에서는 필요없음!!!
//사용안함!!! 일단 호출은 안하도록 한다!
template<typename DTYPE> void text8<DTYPE>::MakeSkipGramLabelData(){

      for (int i = 0; i < m_numOfInput; i++) {

          m_aaLabel[i] = new DTYPE[1];
          //positive sample
          m_aaLabel[i][0] = (DTYPE)0;       //positive sample은 항상 맨 처음
      }
}


//ACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAY

// m_numOfInput = NUMOFWORD / 4;
// 이거 index로 했는지 확인하기!!! 그리고 없는 단어는 pass하도록!!!
template<typename DTYPE> void text8<DTYPE>::MakeAccuracyData(){

      int index = 0;
      int w0 = 0, w1 = 0, w2 = 0, w3 = 0;

      for (int i = 0; i < m_numOfInput; i++) {

          m_aaInput[i] = new DTYPE[m_dimOfInput];
          m_aaLabel[i] = new DTYPE[m_dimOfLabel];

          w0 = word2index(wordTextData[index++]);
          w1 = word2index(wordTextData[index++]);
          w2 = word2index(wordTextData[index++]);
          w3 = word2index(wordTextData[index++]);

          //std::cout<<w0<<index2word(w0)<<" "<<w1<<index2word(w1)<<" "<<w2<<index2word(w2)<<" "<<w3<<index2word(w3)<<'\n';

          //word2index에 없는 단어는 pass하기!
          //-1인 경우 pass하고 data 개수 줄이기!
          if(w0 == -1 || w1 == -1 || w2 == -1 || w3 == -1 ){
              m_numOfInput--;
              //std::cout<<"없는 단어!"<<'\n';
              continue;
          }

          //값 넣기
          m_aaInput[i][0] = w0;
          m_aaInput[i][1] = w1;
          m_aaInput[i][2] = w2;
          m_aaLabel[i][0] = w3;

      }

      std::cout<<"최종 m_numOfInput 개수 : "<<m_numOfInput<<'\n';

}
//ACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAYACCURAY




template<typename DTYPE> int text8<DTYPE>::word2index(string str){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==str){
            return index;
        }
    }
    return -1;
}

template<typename DTYPE> string text8<DTYPE>::index2word(int index){

    return vocab[index];
}

//이거는 필요한지 모르겠음....
template<typename DTYPE> int text8<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int text8<DTYPE>::GetWordNum(){
    return word_num;
}

template<typename DTYPE> string* text8<DTYPE>::GetVocab(){
    return vocab;
}

template<typename DTYPE> int text8<DTYPE>::GetVocabSize(){
    return vocab_size;
}

template<typename DTYPE> int text8<DTYPE>::GetInputDim(){
    return m_dimOfInput;
}

template<typename DTYPE> int text8<DTYPE>::GetLabelDim(){
    return m_dimOfLabel;
}



template<typename DTYPE> std::vector<Tensor<DTYPE> *> *text8<DTYPE>::GetData(int idx) {
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

      for (int i = 0; i < m_dimOfInput; i++) {
          //이거는 전체 단어의 개수 안 맞춰주면 이렇게 됨!!!
          if(m_aaInput[idx][i]==-1)
              std::cout<<'\n'<<"****************************************************************************************음수존재..."<<'\n';
          (*input)[i] = m_aaInput[idx][i];
      }

      //(*label)[ (int)m_aaLabel[idx][0] ] = 1.f;
      (*label)[0] = 1.f;

      result->push_back(input);
      result->push_back(label);

      return result;
}

template<typename DTYPE> int text8<DTYPE>::GetLength() {
        return m_numOfInput;
}
