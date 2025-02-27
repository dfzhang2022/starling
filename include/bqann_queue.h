#pragma once
#ifndef BQANN_QUEUE
#define BQANN_QUEUE

#include "concurrentqueue.h"


#include <unordered_map>
#include <vector>
#include <glog/logging.h>





namespace bqann {

  template<typename T>
  class BaseQueue {
   public:
    virtual void push(T const& new_item) = 0;

    virtual bool try_pop(T& return_item) = 0;

    virtual bool empty() = 0;
  };

  template<typename T>
  class NormalQueue : public BaseQueue<T> {
   public:
    NormalQueue() = default;
    ~NormalQueue() = default;
    void push(T const& new_item) {
      this->inner_queue.enqueue(new_item);
      return;
    }
    bool try_pop(T& return_item) {
      return this->inner_queue.try_dequeue(return_item);
    }
    bool empty() {
      return this->inner_queue.size_approx() == 0;
    }

   private:
    moodycamel::ConcurrentQueue<T> inner_queue;
  };

  template<typename T>
  class WrrQueue : public BaseQueue<T> {
   public:
    WrrQueue() {
      this->weights = {1};
      weighted_queues.resize(1);
      this->weight_q_map[1] = &weighted_queues[0];
    }
    WrrQueue(const std::vector<uint32_t>& weights) {
      this->weights = weights;
      std::sort(this->weights.begin(), this->weights.end(), std::greater<int>());
      queue_size = this->weights.size();
      this->weighted_queues.resize(queue_size);
      for (size_t i = 0; i < queue_size; i++) {
        this->weight_q_map[this->weights[i]] = &weighted_queues[i];
      }

      std::string log_str = "weights:{";
      for (size_t i = 0; i < queue_size; i++) {
        log_str = log_str + std::to_string(this->weights[i]) + ",";
      }
      LOG(INFO)<<log_str + "}";
    }
    ~WrrQueue() = default;

    void push(T const& new_item);
    void push_with_weight(T const& new_item, int weight = 1);
    bool try_pop(T& return_item);
    bool empty();

   private:
    size_t                last_pop_queue_index = 0;
    size_t                last_queue_cnt = 0;
    size_t queue_size=0;
    std::vector<uint32_t> weights;
    std::unordered_map<uint32_t, moodycamel::ConcurrentQueue<T>*> weight_q_map;
    std::vector<moodycamel::ConcurrentQueue<T>> weighted_queues;
  };
}  // namespace bqann


#endif // BQANN_QUEUE