/*
 * @Author: dfzhang dfzhang@ruc.edu.cn
 * @Date: 2025-03-01 05:53:15
 * @LastEditors: dfzhang dfzhang@ruc.edu.cn
 * @LastEditTime: 2025-03-01 09:19:02
 * @FilePath: /starling/include/query_scheduler.h
 * @Description: Using for each worker thread/coroutine to get next valid
 * query_id to execution.
 */
#include "concurrentqueue.h"

class QueryScheduler {
 public:
  QueryScheduler() = default;
  ~QueryScheduler() = default;

  void push_query(int q_id, int weight = 1) {
    this->query_id_queue.enqueue({q_id,weight});
  }
  bool get_next(int& next_id) {
    std::pair<int,int> tmp_pair;
    bool res = query_id_queue.try_dequeue(tmp_pair);
    if(res){
      next_id = tmp_pair.first;
    }
    return res;
  }

 private:
  moodycamel::ConcurrentQueue<std::pair<int,int>> query_id_queue;
};

class BinaryWeightedQueryScheduler {
 public:
  BinaryWeightedQueryScheduler() = default;
  ~BinaryWeightedQueryScheduler() = default;

  void push_query(int q_id, int weight = 1) {
    if (weight > 0) {
      this->high_weight_queue.enqueue({q_id, weight});
    } else {
      this->low_weight_queue.enqueue({q_id, weight});
    }
  }
  bool get_next(int& next_id, bool is_high_weight = false) {
    std::pair<int, int>                               tmp_pair;
    moodycamel::ConcurrentQueue<std::pair<int, int>>* tmp_q_ptr;
    tmp_q_ptr = is_high_weight ? &high_weight_queue : &low_weight_queue;
    bool res = tmp_q_ptr->try_dequeue(tmp_pair);
    if (res) {
      next_id = tmp_pair.first;
    }
    return res;
  }

 private:
  moodycamel::ConcurrentQueue<std::pair<int, int>> low_weight_queue;
  moodycamel::ConcurrentQueue<std::pair<int, int>> high_weight_queue;
};