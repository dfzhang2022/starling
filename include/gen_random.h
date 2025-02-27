#include <iostream>
#include <random>

class BinaryRandomGenerator {
 public:
  BinaryRandomGenerator(double prob_1 = 0.2)
      : rng(std::random_device{}()), prob(prob_1), distribution(0, 99) {
  }  // 生成 0-99 的整数

  int generate() {
    return (distribution(rng) < 100 * prob) ? 1 : 0;
  }

 private:
  double                             prob = 0;
  std::mt19937                       rng;  // 高性能随机数引擎
  std::uniform_int_distribution<int> distribution;
};