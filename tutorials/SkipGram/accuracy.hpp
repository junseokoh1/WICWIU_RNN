#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"


#define NUMOFACCWORD         78176      //questions-words.txt에서 단어의 개수!

using namespace std;

template<typename DTYPE>
class accuracy : public Dataset<DTYPE> {
private:

    //textData에 있던 변수들!!!
    string* vocab;          //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;         //파일에서 읽어오기!

    string* wordTextData;   //strtok를 사용하면 원래 data가 바뀌어서 추가한거!

    int vocab_size;         //반복없는 단어의 개수
    int text_length;        // 이거는 char 개수...     //나중에 fastText에서 필요할 수도 있을거 같아서 남겨둠!!!
    int word_num;           //단어의 개수

    //word2vec.hpp에 있던 거!!!
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;             //input data의 개수!!!

    int m_dimOfInput;
    int m_dimOfLabel;


public:
    accuracy(string File_Path, string* pVocab, int pVocab_size) {
          vocab = NULL;
          TextData = NULL;
          wordTextData = NULL;

          vocab_size = 0;           //이 값을 설정을 못해줌!!!.....
          text_length = 0;
          word_num=0;

          //word2vec.hpp에 있던거!
          m_aaInput = NULL;
          m_aaLabel = NULL;

          m_numOfInput = 0;

          m_dimOfInput = 0;
          m_dimOfLabel = 0;

          Alloc(File_Path, pVocab, pVocab_size);
    }

    virtual ~accuracy() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string pTextPath, string* pVocab, int pVocab_size);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

    void                                  MakeAccuracyData();

    int                                   word2index(string str);

    string                                index2word(int index);

    int                                   GetTextLength();

    int                                   GetWordNum();

    int                                   GetVocabSize();

    int                                   GetInputDim();
    int                                   GetLabelDim();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void accuracy<DTYPE>::Alloc(string pTextPath, string *pVocab, int pVocab_size) {

    std::cout<<"----------------------------accuracy::Alloc 함수 호출----------------------"<<'\n';

    vocab        = pVocab;
    vocab_size   = pVocab_size;
    wordTextData = new string[NUMOFACCWORD];

    m_numOfInput = NUMOFACCWORD / 4;
    m_dimOfInput = 3;
    m_dimOfLabel = 1;

    m_aaInput = new DTYPE *[m_numOfInput];
    m_aaLabel = new DTYPE *[m_numOfInput];

    FileReader(pTextPath);

    MakeVocab();

    MakeAccuracyData();

}


template<typename DTYPE> void accuracy<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void accuracy<DTYPE>::FileReader(string pFile_Path) {
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
      }

      Eliminate(TextData, '\n');
      replace(TextData, '\r', ' ');
    }

    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    fin.close();
}

template<typename DTYPE> void accuracy<DTYPE>::MakeVocab(){

    char* token = strtok(TextData, " ");    //단어 하나씩 가져오기 위한 변수

    int word_count =0;

    while(token != NULL){

        wordTextData[word_count] = token;

        token = strtok(NULL, " ");
        //단어 개수
        word_count++;
    }

    word_num = word_count;

}


// m_numOfInput = NUMOFACCWORD / 4;
// 이거 index로 했는지 확인하기!!! 그리고 없는 단어는 pass하도록!!!
template<typename DTYPE> void accuracy<DTYPE>::MakeAccuracyData(){

      int textIndex = 0, inputIndex = 0, miss = 0;
      int w0 = 0, w1 = 0, w2 = 0, w3 = 0;

      for (int i = 0; i < m_numOfInput; i++) {

          w0 = word2index(wordTextData[textIndex++]);
          w1 = word2index(wordTextData[textIndex++]);
          w2 = word2index(wordTextData[textIndex++]);
          w3 = word2index(wordTextData[textIndex++]);

          //word2index에 없는 단어는 pass하기!
          if(w0 == -1 || w1 == -1 || w2 == -1 || w3 == -1 ){
              miss++;
              continue;
          }

          // std::cout<<w0<<index2word(w0)<<" "<<w1<<index2word(w1)<<" "<<w2<<index2word(w2)<<" "<<w3<<index2word(w3)<<'\n';
          m_aaInput[inputIndex] = new DTYPE[m_dimOfInput];
          m_aaLabel[inputIndex] = new DTYPE[m_dimOfLabel];

          m_aaInput[inputIndex][0] = w0;
          m_aaInput[inputIndex][1] = w1;
          m_aaInput[inputIndex][2] = w2;
          m_aaLabel[inputIndex][0] = w3;

          std::cout<<index2word(w0)<<" "<<index2word(w1)<<" "<<index2word(w2)<<" "<<index2word(w3)<<'\n';

          inputIndex++;
      }

      m_numOfInput -= miss;

      std::cout<<"최종 m_numOfInput 개수 : "<<m_numOfInput<<"  "<<inputIndex<<'\n';

}

template<typename DTYPE> int accuracy<DTYPE>::word2index(string str){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==str){
            return index;
        }
    }
    return -1;
}

template<typename DTYPE> string accuracy<DTYPE>::index2word(int index){

    return vocab[index];
}

//이거는 필요한지 모르겠음....
template<typename DTYPE> int accuracy<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int accuracy<DTYPE>::GetWordNum(){
    return word_num;
}

template<typename DTYPE> int accuracy<DTYPE>::GetVocabSize(){
    return vocab_size;
}

template<typename DTYPE> int accuracy<DTYPE>::GetInputDim(){
    return m_dimOfInput;
}

template<typename DTYPE> int accuracy<DTYPE>::GetLabelDim(){
    return m_dimOfLabel;
}


//여기를 수정해야 label값이 원하는 형태로 변경됨!!!
template<typename DTYPE> std::vector<Tensor<DTYPE> *> *accuracy<DTYPE>::GetData(int idx) {
      std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

      Tensor<DTYPE> *input = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfInput);
      Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, m_dimOfLabel);

      for (int i = 0; i < m_dimOfInput; i++) {
          (*input)[i] = m_aaInput[idx][i];
      }

      for (int i = 0; i < m_dimOfLabel; i++) {
          (*label)[i] = m_aaLabel[idx][i];
      }

      result->push_back(input);
      result->push_back(label);

      return result;
}

template<typename DTYPE> int accuracy<DTYPE>::GetLength() {
        return m_numOfInput;
}
