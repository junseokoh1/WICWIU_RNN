#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include <time.h>   //난수 생성위해 추가!

#include "../../WICWIU_src/Tensor.hpp"
//#include "../../WICWIU_src/DataLoader.hpp"        // 왜 추가한거지?   Dataset때문에 추가한거 같음... 추측임

#define NUMOFWORD       71291   //text8에서 단어의 개수!

using namespace std;

enum OPTION {
    ONEHOT,
    CBOWMODE,
    SKIPGRAM
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
class text8 {
private:

    //textData에 있던 변수들!!!
    string* vocab;          //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;         //파일에서 읽어오기!

    string* wordTextData;   //strtok를 사용하면 원래 data가 바뀌어서 추가한거!

    int vocab_size;         //반복없는 단어의 개수
    int text_length;        // 이거는 char 개수...
    int word_num;           //단어의 개수

    //각 단어가 몇 번 나왔는지는 없음...! 이거 배열을 하나 더 만들어서 가능할듯!!!

    OPTION option;

    //word2vec.hpp에 있던 거!!!
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;             //input data의 개수!!!
    int m_numOfLabel;
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

          input = NULL;
          label = NULL;

          option = pOption;

          //word2vec.hpp에 있던거!
          m_aaInput = NULL;
          m_aaLabel = NULL;

          m_numOfInput = 0;
          m_numOfLabel = 0;
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

    //ONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOT
    //이거 2개는 그냥 다음 단어 예측하도록 만든거
    // void                                  MakeInputData();
    // void                                  MakeLabelData();
    // //ONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOT
    //
    // //CBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOW
    // void                                  MakeCBOWInputData();
    // void                                  MakeCBOWLabelData();
    //CBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOW

    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM
    void                                  MakeSkipGramInputData();
    void                                  MakeSkipGramLabelData();
    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM

    int                                   word2index(string str);

    string                                index2word(int index);

    int                                   GetTextLength();

    int                                   GetWordNum();

    int                                   GetVocabLength();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void text8<DTYPE>::Alloc(string File_Path) {

    vocab = new string[NUMOFWORD];
    wordTextData = new string[NUMOFWORD];


    //word2vec.hpp
    m_window   =  window;
    m_negative = negative;
    // m_numOfInput = NUMOFWORD * (m_window - 1);     //sos eos 추가한거!
    // m_numOfLabel = NUMOFWORD * (m_window - 1);     //sos eos 추가한거!
    m_numOfInput = NUMOFWORD * (m_window - 1) - 2;     //sos eos 제외한 버전
    m_numOfLabel = NUMOFWORD * (m_window - 1) - 2;     //sos eos 제외한 버전
    m_dimOfInput = m_negative + 2;      //center word와 context neighbor word 2개 추가!
    m_dimOfLabel = m_negative + 1;      //positive sample에 대한 답

    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfLabel];

    //File_Reader
    FileReader(File_Path);

    //make_vocab
    MakeVocab();

    // if(option==ONEHOT){
    //      MakeInputData();
    //      MakeLabelData();
    //  }
    //  else if(option == CBOWMODE){
    //       MakeCBOWInputData();
    //       MakeCBOWLabelData();
    // }
    

}


template<typename DTYPE> void text8<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void text8<DTYPE>::FileReader(string pFile_Path) {
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

      //소문자로 변환
      for(int i=0; i<text_length; i++){
          TextData[i] = tolower(TextData[i]);
          std::cout<<TextData[i];
      }

      // std::cout<<"제거 전 text 길이 : "<<text_length<<'\n';
      // std::cout<< strlen(TextData)<<'\n';

      //빈공간 다 없애고 sp만 남기기
      Eliminate(TextData, '\n');
      Eliminate(TextData, ':');
      Eliminate(TextData, ',');
      Eliminate(TextData, '.');
      Eliminate(TextData, ';');
      Eliminate(TextData, '?');
      Eliminate(TextData, '!');
      replace(TextData, '\r', ' ');
    }

    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    fin.close();
}

template<typename DTYPE> void text8<DTYPE>::MakeVocab(){

    int flag = 0;
    char* token = strtok(TextData, " ");    //단어 하나씩 가져오기 위한 변수

    int word_count = 0;

    while(token != NULL){

        wordTextData[word_count] = token;

        //std::cout<<token<<'\n';

        //중복확인하기
        for(int i=0; i<vocab_size; i++){
            if(vocab[i] == token){
                flag = 1;
                // std::cout<<"중복된 단어 : "<<token<<'\n';
            }
        }

        //중복이 아니라면
        if(flag == 0){
            vocab[vocab_size] = token;
            vocab_size++;
        }

        token = strtok(NULL, " ");
        //단어 개수
        word_count++;
        flag = 0;
    }

    //SOS하고 EOS 추가하려면 여기서!!! 그리고 vocab수하고 파일에 있는 단어수도 증가시키기???
    //sos : "<s>"    eos : "<\s>"
    vocab[vocab_size] = "<s>";
    vocab_size ++;
    vocab[vocab_size] = "<\s>";       //이거 가능?
    vocab_size ++;


    sort(vocab, vocab+vocab_size-1);

    word_num = word_count;


    // //여기서 부터는 확인해보려고 찍어 보는거!!!
    // std::cout<<"단어 개수 : "<<word_num<<'\n';
    //
    // //wordTextData 이게 잘 되어 있는가 확인하기
    // for(int i=0; i<word_num; i++){
    //     std::cout<<wordTextData[i]<<" ";
    // }
    //
    // std::cout<<'\n';
    //
    // std::cout<<"vocab size(중복없는 단어 개수) : "<<vocab_size<<'\n';
    // for(int i=0; i<vocab_size; i++){
    //     std::cout<<vocab[i]<<" ";
    // }
    //
    // std::cout<<'\n';
}


//이거는... sos하고 eos없는 버전이라고 생각하자!!!
//일단 subsampling of frequent words - 이거는 일단 제외하고 하겠음!!
// center  context  non-context  non-context
template<typename DTYPE> void text8<DTYPE>::MakeSkipGramInputData(){

      srand(time(NULL));

      int flag = 0;   // window size에 따라 center index를 바꿔주기 위해!

      //context index에 대한 정보를 담고있는 배열  flag로 접근!
      int contextOffset[m_window-1];
      for(int i=0; i<m_window%2; i=i+2;){
          contextOffset[i] = i+1;
          contextOffset[i+1] = -(i+1);
      }

      int centerIndex = 0;
      int non_contextIndex=0;


      for (int i = 0; i < m_numOfInput; i++) {

            m_aaInput[i] = new DTYPE[m_dimOfInput];

            //center word
            m_aaInput[i][0] = word2index(wordTextData[centerIndex]);

            //context word 주고...
            //양 끝에 처리해주기!!!
            if(centerIndex+contextOffset[flag] < 0){

                m_aaInput[i][1] = word2index("<s>");

            }else if(centerIndex+contextOffset[flag] > word_num){

                m_aaInput[i][1] = word2index("<\s>");

            }else{

                m_aaInput[i][1] = word2index(wordTextData[ centerIndex+contextOffset[flag] ]);

            }


            //negative 주고....
            for (int d = 2; d < m_dimOfInput; d++) {
                //여기서 이제 random으로 생성해주고!!!
                contextIndex = rand()%word_num;

                //겹치면 안겹치도록!!! center뿐만 아니라 context에 대해서도 처리가 필요!!!
                if(contextIndex == centerIndex)
                    continue;

                m_aaInput[i][d] = word2index(wordTextData[i]);
            }

            flag++;

            if(flag == m_window-1){
                centerIndex++;
                flag = 0;
            }

      }

}

template<typename DTYPE> void text8<DTYPE>::MakeSkipGramLabelData(){

      for (int i = 0; i < m_numOfLabel; i++) {

          m_aaLabel[i] = new DTYPE[m_dimOfLabel];
          //positive sample
          pLabel[i][0] = 1;
          //negative sample
          for(int j=1; j< m_dimOfLabel; j++)
              pLabel[i][j] = 0;
      }
}




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

template<typename DTYPE> int text8<DTYPE>::GetVocabLength(){
    return vocab_size;
}



template<typename DTYPE> std::vector<Tensor<DTYPE> *> *text8<DTYPE>::GetData(int idx) {
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *image = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

      //DIMOFMNISTIMAGE = 784로 설정되어 있음
      for (int i = 0; i < m_dimOfInput; i++) {
          (*image)[i] = m_aaInput[idx][i];              //tensor에 대해 [] operator를 지정해줘서 바로 m_aLongArray배열에 저장
      }

      //DIMOFMNISTLABEL = 10으로 설정되어 있음!, 그래서 이건 one-hot으로!!!!
      (*label)[ (int)m_aaLabel[idx][0] ] = 1.f;

      result->push_back(image);         //push_back 함수는 vector에 있는 함수인가?
      result->push_back(label);

      return result;
}

//이부분 다시 수정하기!
template<typename DTYPE> int text8<DTYPE>::GetLength() {
        return m_numOfInput;
}
