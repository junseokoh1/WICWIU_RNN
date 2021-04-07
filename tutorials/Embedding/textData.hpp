#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string.h>

#include "../../WICWIU_src/Tensor.hpp"
//#include "../../WICWIU_src/DataLoader.hpp"        // 왜 추가한거지?   Dataset때문에 추가한거 같음... 추측임

using namespace std;

enum OPTION {
    ONEHOT,
    CBOWMODE
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
class textData {
private:

    string* vocab;          //이제 단어들을 갖고 있어야 하니깐!!!, 중복을 제거한 단어!
    char* TextData;         //파일에서 읽어오기!

    string* wordTextData;   //strtok를 사용하면 원래 data가 바뀌어서 추가한거!

    int vocab_size;         //반복없는 단어의 개수
    int text_length;        // 이거는 char 개수...
    int word_num;           //단어의 개수

    Tensor<DTYPE>* input;
    Tensor<DTYPE>* label;

    OPTION option;

    int VOCAB_LENGTH;

public:
    textData(string File_Path, int vocab_length, OPTION pOption) {
          vocab = NULL;
          TextData = NULL;
          wordTextData = NULL;

          vocab_size = 0;
          text_length = 0;
          word_num=0;

          input = NULL;
          label = NULL;

          option = pOption;

          VOCAB_LENGTH = vocab_length;

          Alloc(File_Path);
    }

    virtual ~textData() {
        Delete();
    }

    //왜 굳이 virtual인거지?
    void                                  Alloc(string File_Path);

    void                                  Delete();

    void                                  FileReader(string pFile_Path);
    void                                  MakeVocab();

                                                                                //make 함수들!!! 전부다 float로 해버림!!! 이거 DTYPE으로 수정필요함 !!!!!!!!!!!!!!!!

    //ONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOT
    //이거 2개는 그냥 다음 단어 예측하도록 만든거
    void                                  MakeInputData();
    void                                  MakeLabelData();
    //ONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOT

    //CBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOW
    void                                  MakeCBOWInputData();
    void                                  MakeCBOWLabelData();
    //CBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOW

    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM
    void                                  MakeSkipGramInputData();
    void                                  MakeSkipGramLabelData();
    //SKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAMSKIPGRAM

    int                                   char2index(string str);

    string                                index2char(int index);

    Tensor<DTYPE>*                        GetInputData();

    Tensor<DTYPE>*                        GetLabelData();

    int                                   GetTextLength();

    int                                   GetWordNum();

    int                                   GetVocabLength();

    //virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    //virtual int                           GetLength();

};

template<typename DTYPE> void textData<DTYPE>::Alloc(string File_Path) {

    vocab = new string[VOCAB_LENGTH];
    wordTextData = new string[VOCAB_LENGTH];

    //File_Reader
    FileReader(File_Path);

    //make_vocab
    MakeVocab();

    if(option==ONEHOT){
         MakeInputData();
         MakeLabelData();
     }
     else if(option == CBOWMODE){
          MakeCBOWInputData();
          MakeCBOWLabelData();
      }
}


template<typename DTYPE> void textData<DTYPE>::Delete() {
    delete []vocab;
    delete []TextData;
}

template<typename DTYPE> void textData<DTYPE>::FileReader(string pFile_Path) {
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
      //space가 2개 있는 이유가 cr+lf cr+lf가 연속으로 존재해서 생김...
      //replace(TextData, '  ', ' ');

      std::cout<<'\n'<<TextData<<'\n';
    }

    //제거했으니깐 길이 다시 설정해주기
    text_length = strlen(TextData);     //strlen원리가 NULL를 찾을 때 까지여서 마지막에 NULL이 자동으로 추가된거 같음!

    //마지막에 NULL 추가해주기!   -> 이거 eliminate에서 마지막에 있는 NULL까지 이동시켜줘서 괜춘!

    // std::cout<<"제거 후 text 길이 : "<<text_length<<'\n';

    //std::cout<<TextData[text_length]<<'\n';

    fin.close();
}

template<typename DTYPE> void textData<DTYPE>::MakeVocab(){

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

    sort(vocab, vocab+vocab_size-1);

    //출력해보기
    // for(int i=0; i<vocab_size; i++)
    //     std::cout<<i<<"번째 vocab : "<<vocab[i]<<'\n';

    word_num = word_count;


    //여기서 부터는 확인해보려고 찍어 보는거!!!
    std::cout<<"단어 개수 : "<<word_num<<'\n';

    //wordTextData 이게 잘 되어 있는가 확인하기
    for(int i=0; i<word_num; i++){
        std::cout<<wordTextData[i]<<" ";
    }

    std::cout<<'\n';

    std::cout<<"vocab size(중복없는 단어 개수) : "<<vocab_size<<'\n';
    for(int i=0; i<vocab_size; i++){
        std::cout<<vocab[i]<<" ";
    }

    std::cout<<'\n';
}


//ONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOTONEHOT
template<typename DTYPE> void textData<DTYPE>::MakeInputData(){

         int* onehotvector = new int[vocab_size];

         input = new Tensor<DTYPE>(word_num, 1, 1, 1, vocab_size);


         for(int i=0; i<word_num; i++){

              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
              }
         }

}

template<typename DTYPE> void textData<DTYPE>::MakeLabelData(){

        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(word_num, 1, 1, 1, vocab_size);

        for(int i=0; i<word_num; i++){

            //마지막 data
            if(i==word_num-1){
                  MakeOneHotVector(onehotvector, vocab_size, vocab_size-1);
                  for(int j=0; j<vocab_size; j++){
                      (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
                  }
              continue;
            }

            MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*label)[Index5D(label->GetShape(), i, 0, 0, 0, j)] = onehotvector[j];
            }
        }

}





//CBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOWCBOW

template<typename DTYPE> void textData<DTYPE>::MakeCBOWInputData(){

        std::cout<<"------------------CBOWInput-------------"<<'\n';

         int* onehotvector = new int[vocab_size];

         input = new Tensor<DTYPE>(1, word_num-2, 1, 1, vocab_size*2);                //batch로 형태를 바꿈!!!

         for(int i=0; i<word_num-2; i++){

              //앞쪽에 해당하는 context input1
              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), 0, i, 0, 0, j)] = onehotvector[j];
              }

              //앞쪽에 해당하는 context input1
              MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i+2]));
              for(int j=0; j<vocab_size; j++){
                  (*input)[Index5D(input->GetShape(), 0, i, 0, 0, vocab_size+j)] = onehotvector[j];
              }
         }

}

template<typename DTYPE> void textData<DTYPE>::MakeCBOWLabelData(){

        std::cout<<"------------------CBOWLabel-------------"<<'\n';

        int* onehotvector = new int[vocab_size];

        label = new Tensor<float>(1, word_num-2, 1, 1, vocab_size);                 //batch로 형태를 바꿈!!!

        for(int i=0; i<word_num-2; i++){

             // std::cout<<"index : "<<i<<'\n';
             // std::cout<<"index에 해당하는 단어 : "<<wordTextData[i+1]<<'\n';

            MakeOneHotVector(onehotvector, vocab_size, char2index(wordTextData[i+1]));
            for(int j=0; j<vocab_size; j++){
                (*label)[Index5D(label->GetShape(), 0, i, 0, 0, j)] = onehotvector[j];
            }
        }

}



template<typename DTYPE> int textData<DTYPE>::char2index(string str){

    for(int index=0; index<vocab_size; index++){
        if(vocab[index]==str){
            //std::cout<<index2char(index)<<'\n';               //여기 출력!!!!!!!!!!!!!!!!!!
            return index;
        }
    }
    //std::cout<<"못찾음"<<'\n';
    return -1;
}

template<typename DTYPE> string textData<DTYPE>::index2char(int index){

    return vocab[index];
}

template<typename DTYPE> Tensor<DTYPE>* textData<DTYPE>::GetInputData(){

    return input;
}

template<typename DTYPE> Tensor<DTYPE>* textData<DTYPE>::GetLabelData(){
    return label;
}

template<typename DTYPE> int textData<DTYPE>::GetTextLength(){
    return text_length;
}

template<typename DTYPE> int textData<DTYPE>::GetWordNum(){
    return word_num;
}

template<typename DTYPE> int textData<DTYPE>::GetVocabLength(){
    return vocab_size;
}
