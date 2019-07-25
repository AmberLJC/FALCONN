//
// Created by 刘嘉晨 on 2019-06-22.
//
#include <omp.h>



#include <numeric>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#include <string.h>
#include <cstdlib>
#include <string>
#include<fstream>
#include "float.h"
#include<sstream>

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    vector<size_t> idx(v.size());
    //使用iota对向量赋0~？的连续值
    iota(idx.begin(), idx.end(), 0);

    // 通过比较v的值对索引idx进行排序
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2)
         {return v[i1] < v[i2];});
    //increasing order
    return idx;
}
//./main dense1k 1000 100

int main(int argc, char *argv[])
//int main()
{
    const int NUM = std::atoi( argv[2]);
    const int Q=1000;
    const int K=100;
    const int DIM= atoi(argv[3]);
    /*
    char a[20];//定义字符数组a,b
    strcpy(a,argv[1]);//将b中
    const char *LEARN =strcat(a,"_learn.txt");
    strcpy(a,argv[1]);
    const char *GT =strcat(a,"_gt.txt");

    strcpy(a,argv[1]);
    const char *BASE =strcat(a,"_base.txt");
    strcpy(a,argv[1]);
    const char *QUERY =strcat(a,"_query.txt");
*/
    char IN[20];
    strcpy(IN,argv[1]);
    const char *a =strcat(IN,".txt");

    vector< vector<float> > data(NUM,vector<float> (DIM+1));

    cout<<"Generating input "<< IN <<" for faiss, "<< NUM <<" data with "<<DIM<<" dimension."<<endl;


    ifstream infile;//定义读取文件流，相对于程序来说是in
    cout<<"Start reading !"<<endl;
    infile.open( IN  );//打开文件

    for (int i = 0; i < NUM; i++)//定义行循环
    {data[i][0]=i;

        for (int j = 1; j < 1+ DIM; j++)//定义列循环
        {
            infile >> data[i][j];//读取一个值（空格、制表符、换行隔开）就写入到矩阵中，行列不断循环进行
            //cout << data[i][j]<< " , ";
        }
    }
    infile.close();//读取完成之后关闭文件
    cout<<"finish reading !"<<endl;
    for (int j = 0; j < DIM+1; j++)//定义列循环
    {
        cout << data[0][j]<< " ,";//以下代码是用来验证读取到的值是否正确
    }
    cout<<endl;

    char BASE[20];//定义字符数组a,b
    strcpy(BASE,argv[1]);//将b中
    strcat(BASE,"_base.txt");
    ofstream outfile (BASE);
    if(!outfile.is_open())
    {
        cout<<" the file open fail"<<endl;
        exit(1);
    }
    for(int i=0;i<NUM;i++)
    {
        for(int j=0;j<DIM+1;j++)
        {
            outfile<<data[i][j]<<" ";
        }
        outfile<<"\r\n";
    }
    cout<<"finish writing to "<<BASE<<endl;
    outfile.flush();
    outfile.close();
    char LEARN[20];//定义字符数组a,b
    strcpy(LEARN,argv[1]);//将b中
    strcat(LEARN,"_learn.txt");

    ofstream outfile2 (LEARN);
    if(!outfile2.is_open())
    {
        cout<<" the file open fail"<<endl;
        exit(1);
    }

    int str = int(NUM/10);
    int end = int(NUM/5) ;
    for(int i= str ;i< end ;i++)
    {
        for(int j=0;j<DIM+1;j++)
        {
            outfile2<<data[i][j]<<" ";
        }
        outfile2<<"\r\n";
    }
    cout<<"finish writing to "<<LEARN<<endl;

    outfile2.close();
    outfile2.flush();

    char QUERY[20];//定义字符数组a,b
    strcpy(QUERY,argv[1]);//将b中
    strcat(QUERY,"_query.txt");

    ofstream outfile3(QUERY);
    if(!outfile3.is_open())
    {
        cout<<QUERY<< " file open fail"<<endl;
        exit(1);
    }
    for(int i = 0 ;i < Q ; i++)
    {
        for(int j = 0 ;j < DIM+1 ; j++ )
        {
            outfile3<<data[i][j]<<" ";
        }
        outfile3<<"\r\n";
    }
    cout<<"finish writing to "<<QUERY<<endl;

    outfile3.close();
    outfile3.flush();


    char GT[20];//定义字符数组a,b
    strcpy(GT,argv[1]);//将b中
    strcat(GT,"_gt.txt");

int const THREAD_NUM = 60 ; 
    ofstream outfile4 (GT);
    if(!outfile4.is_open())
    {
        cout<<GT<< " file open fail"<<endl;
        exit(1);
    }
    #pragma omp parallel for num_threads(THREAD_NUM)
    for(int i = 0 ;i < Q ; i++){// 1000 query

       // vector<int>  dis_list(NUM);
       float min_dis =DBL_MAX;
        int min_idx=i;
#pragma omp parallel for num_threads(THREAD_NUM)
 	for(int j = 0 ;j < NUM ; j++ ){// linear scan
            float dis=0;

//#pragma omp parallel for num_threads(THREAD_NUM)

	    for(int d = 1 ; d< DIM+1 ; d++) {

                dis+= ( data[j][d]-data[i][d])*( data[j][d]-data[i][d]);
                if (dis > min_dis) break;
            }
            if( dis < min_dis && dis>0){

                min_idx=j;
                min_dis=dis;
            }

           // dis_list[j]=dis;
        }

      //  vector<size_t> res(NUM);
        //res= sort_indexes (dis_list);
        outfile4<< 0 <<" "<<min_idx<<" ";
        for(int j = 2 ;j < K+1 ; j++ )
        {

            outfile4<< 0 <<" ";

        }
        outfile4<<"\r\n";

        if(i%100==0) cout<<"Find 10% 1nn groundtruth : "<<min_idx<<endl;
    }

    outfile4.close();


    return 0;




}



