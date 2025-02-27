#include "bqann_queue.h"
#include <iostream>

namespace bqann {
     
    template<typename T>
    void WrrQueue<T>::push(T const& new_item){
        this->weight_q_map[1]->enqueue(new_item);
    }
    template<typename T>
    void WrrQueue<T>::push_with_weight(T const& new_item, int weight){
        if(this->weight_q_map.find(weight) == weight_q_map.end()){
            // push to defaut queue
            // this->weight_q_map[1]->enqueue(new_item);
            LOG(ERROR)<<"No valid queue for weight: "<<weight;
        }else{
            // push to specific queue
            this->weight_q_map[weight]->enqueue(new_item);
        }
    }


    template<typename T>
    bool WrrQueue<T>::try_pop(T& return_item){
        // TODO 添加各个队列的WRR调度算法
        size_t empty_cnt =0;
        while(!(weighted_queues[last_pop_queue_index].size_approx()>0)){
            if(empty_cnt >= queue_size) return false;
            last_pop_queue_index = (last_pop_queue_index + 1) % weights.size();
            last_queue_cnt = 0;
            empty_cnt++;
        }
        
        bool res = weighted_queues[last_pop_queue_index].try_dequeue(return_item);
        // std::cout<<weights[last_pop_queue_index]<<" ";
        last_queue_cnt++;
        if(last_queue_cnt >= weights[last_pop_queue_index]){
            last_pop_queue_index = (last_pop_queue_index + 1) % weights.size();
            last_queue_cnt = 0;
        }
        return res;
    }

    template<typename T>
    bool WrrQueue<T>::empty(){
        // TODO 添加判断各个队列为空的算法
        bool res = false;
        for(size_t k = 0 ;k<weighted_queues.size();k++){
            res = res || weighted_queues[k].size_approx()>0;
        }
        return !res;
    }

    template class WrrQueue<int>;
    template class WrrQueue<char>;
    template class WrrQueue<std::pair<int,int>>;
}