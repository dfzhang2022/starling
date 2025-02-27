#include "bqann_queue.h"
#include "gen_random.h"

#include <glog/logging.h>
#include <iostream>

using namespace bqann;


int main(int argc, char** argv) {
    // TODO: Implement test cases for the WRRQueue class

    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr=1; 

    bqann::NormalQueue<int> queue;
    queue.push(1);
    queue.push(2);
    queue.push(3);

    int value;

    while(!queue.empty()){
        bool res = queue.try_pop(value);
        // std::cout << value;
        LOG(INFO) << value;
        if(!res)break;
    }




    WrrQueue<char>* q_ptr =  nullptr;
    q_ptr = static_cast<WrrQueue<char>*>(new WrrQueue<char>({5,2,3}));
    // q_ptr = static_cast<WrrQueue<char>*>(q_ptr);

    q_ptr->push_with_weight('A', 5);
    q_ptr->push_with_weight('B', 5);
    q_ptr->push_with_weight('C', 5);
    q_ptr->push_with_weight('D', 5);
    q_ptr->push_with_weight('E', 5);
    q_ptr->push_with_weight('F', 5);
    q_ptr->push_with_weight('G', 5);

    q_ptr->push_with_weight('U', 2);
    q_ptr->push_with_weight('V', 2);
    q_ptr->push_with_weight('W', 2);

    q_ptr->push_with_weight('X', 3);
    q_ptr->push_with_weight('Y', 3);

    char ch= 'a';
    while(!q_ptr->empty()){
        // LOG(INFO) << "into loop";
        bool res = q_ptr->try_pop(ch);
        // std::cout << value;
        if(res){
            LOG(INFO) << ch;
        }
    }

    delete q_ptr;

    BinaryRandomGenerator gen;
    for (int i = 0; i < 10; ++i) {
        std::cout << gen.generate() << " ";
    }

    return 0;
}