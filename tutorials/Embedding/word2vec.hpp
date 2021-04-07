#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#define NUMOFWORD       71291   //text8에서 단어의 개수!

using namespace std;

enum OPTION {
    TESTING,
    TRAINING
};

template<typename DTYPE> void Make_INPUT(string pImagePath, DTYPE **pImage) {

}

template<typename DTYPE> void Make_LABEL(int numOfLable, int dimOfLabel, DTYPE **pLabel) {

      for (int i = 0; i < numOfLabel; i++) {

          pLabel[i] = new DTYPE[dimOfLabel];
          //positive sample
          pLabel[i][0] = 1;
          //negative sample
          for(int j=1; j< dimOfLabel; j++)
              pLabel[i][j] = 0;
      }

}

template<typename DTYPE>
class TextDataSet : public Dataset<DTYPE>{
private:

    //Dataset에 필요한 변수들!
    DTYPE **m_aaInput;
    DTYPE **m_aaLabel;

    int m_numOfInput;
    int m_numOfLabel;
    int m_window;
    int m_negative;

    int m_dimOfInput;
    int m_dimOfLabel;

    OPTION m_option;



public:
    TextDataSet(string pTextPath, int window, int negative, OPTION pOPTION) {
        m_aaInput = NULL;
        m_aaLabel = NULL;

        m_numOfInput = 0;
        m_numOfLabel = 0;
        m_window     = 0;
        m_negative   = 0;

        m_dimOfInput = 0;
        m_dimOfLabel = 0;

        m_option = pOPTION;

        Alloc(pTextPath, window, negative);
    }

    virtual ~TextDataSet() {
        Delete();
    }

    virtual void                          Alloc(string pTextPath, int window, int negative);

    virtual void                          Delete();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void TextDataSet<DTYPE>::Alloc(string pTextPath, int window, int negative) {

      //할당해주고
      m_window   =  window;
      m_negative = negative;

      //사이즈 구하기!
      m_numOfInput = NUMOFWORD * (m_window - 1);
      m_numOfLabel = NUMOFWORD * (m_window - 1);
      m_dimOfInput = m_negative + 2;      //center word와 context neighbor word 2개 추가!
      m_dimOfLabel = m_negative + 1;      //positive sample에 대한 답


      if (m_option == TRAINING) {
          m_aaInput = new DTYPE *[m_numOfInput];
          Make_INPUT(pTextPath, m_aaInput);
          m_aaLabel = new DTYPE *[m_numOfLabel];
          Make_LABEL(m_numOfLabel, m_dimOfLabel, m_aaLabel);

      //여기서 부터는 신경 안씀!
      } else if (m_option == TESTING) {
          m_aaInput = new DTYPE *[m_numOfInput];
          Make_INPUT(pImagePath, m_aaInput);
          m_aaLabel = new DTYPE *[m_numOfInput];
          Make_LABEL(m_numOfLabel, m_dimOfLabel, m_aaLabel);
      } else {
          printf("invalid option\n");
          exit(-1);
      }
}

template<typename DTYPE> void TextDataSet<DTYPE>::Delete() {
    if (m_aaInput) {
        for (int i = 0; i < m_numOfInput; i++) {
            if (m_aaInput[i]) {
                delete[] m_aaInput[i];
                m_aaInput[i] = NULL;
            }
        }
        delete m_aaInput;
        m_aaInput = NULL;
    }

    if (m_aaLabel) {
        for (int i = 0; i < m_numOfLabel; i++) {
            if (m_aaLabel[i]) {
                delete[] m_aaLabel[i];
                m_aaLabel[i] = NULL;
            }
        }
        delete m_aaLabel;
        m_aaLabel = NULL;
    }
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *TextDataSet<DTYPE>::GetData(int idx) {
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
template<typename DTYPE> int TextDataSet<DTYPE>::GetLength() {
    // if (m_option == TRAINING) {
        return m_numOfInput;
    // } else if (m_option == TESTING) {
    //     return NUMOFTESTDATA;           //
    // }
    // return 0;
}
