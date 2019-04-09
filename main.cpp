/*******************************************************************************
 * Copyright (C) 2019 Jonas Dann <jonas@dann.io>
 *
 * All rights reserved. Published under the Apache License version 2.0 in the
 * LICENSE file.
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <assert.h>
#include <cmath>

template <
        typename ValueType>
class OnlineSampler {
    class Buffer {
    public:
        size_t weight_;
        std::vector<ValueType> elements_;
        bool sorted_;

        explicit Buffer(size_t k) : weight_(0), sorted_(false), k_(k) {
            elements_.reserve(k);
        }

        bool Put(ValueType value) {
            assert(HasCapacity());
            elements_.emplace_back(value);
            return HasCapacity();
        }

        bool HasCapacity() {
            return elements_.size() < k_;
        }

        bool IsEmpty() {
            return elements_.empty();
        }
    private:
        size_t k_;
    };

public:
    OnlineSampler(size_t b, size_t k) : b_(b), k_(k), current_buffer_(-1), empty_buffers_(b),
            minimum_level_(0) {
        buffers_ = std::vector<Buffer>(b, Buffer(k));
    }

    bool Put(ValueType value) {
        if (current_buffer_ == -1 || !buffers_[current_buffer_].HasCapacity()) {
            New();
        }
        return buffers_[current_buffer_].Put(value) || empty_buffers_ > 0;
    }

    // TODO emitter lambda instead of out_elements
    template <typename Emitter>
    bool Collapse(const Emitter& emit) {
        std::vector<Buffer> level;
        auto level_begin = b_ - level_counters_[minimum_level_];
        for (size_t i = level_begin; i < b_; i++) {
            level.emplace_back(std::move(buffers_[i]));
        }
        size_t weight_sum = 0;
        for (int i = 0; i < level.size(); i++) {
            if (!level[i].sorted_) {
                std::sort(level[i].elements_.begin(), level[i].elements_.end());
                level[i].sorted_ = true;
            }
            weight_sum += level[i].weight_;
        }
        auto target_buffer_index = level_begin;
        buffers_[target_buffer_index].weight_ = weight_sum;
        buffers_[target_buffer_index].sorted_ = true;
        size_t total_index = 0;
        std::vector<size_t> positions(level.size(), 0);
        for (int j = 0; j < k_; j++) {
            size_t target_rank = GetTargetRank(j, buffers_[target_buffer_index].weight_);
            // TODO switch to tournament tree
            ValueType sample;
            bool first = true;
            while (total_index < target_rank) {
                size_t minimum_index = 0;
                while (positions[minimum_index] >= level[minimum_index].elements_.size()) { // empty buffers
                    minimum_index++;
                }
                ValueType minimum = level[minimum_index].elements_[positions[minimum_index]];
                for (int i = minimum_index + 1; i < level.size(); i++) {
                    if (positions[i] < level[i].elements_.size()) { // not empty
                        ValueType value = level[i].elements_[positions[i]];
                        if (value < minimum) {
                            minimum = value;
                            minimum_index = i;
                        }
                    }
                }
                if (first) {
                    first = false;
                } else {
                    emit(sample);
                }
                total_index += level[minimum_index].weight_;
                positions[minimum_index]++;
                sample = minimum;
            }
            buffers_[target_buffer_index].Put(sample);
        }

        empty_buffers_ += level.size() - 1;
        level_counters_[minimum_level_] = 0;
        current_buffer_ = -1;

        minimum_level_++;
        if (minimum_level_ >= level_counters_.size()) {
            level_counters_.emplace_back(1);
        }
        assert(minimum_level_ < level_counters_.size());

        return b_ - empty_buffers_ > 1;
    }

    size_t GetHighestWeightedSamples(std::vector<ValueType>& out_samples) {
        out_samples = buffers_[0].elements_;
        return buffers_[0].weight_;
    }

private:
    size_t b_;
    size_t k_;

    std::vector<Buffer> buffers_;
    std::vector<size_t> level_counters_;
    size_t current_buffer_;
    size_t empty_buffers_;
    size_t minimum_level_;

    void New() {
        current_buffer_ = std::accumulate(level_counters_.begin(),
                level_counters_.end(), (size_t) 0);
        if (empty_buffers_ > 1) {
            minimum_level_ = 0;
        }
        level_counters_[minimum_level_]++;
        buffers_[current_buffer_].weight_ = 1;
        empty_buffers_--;
    }

    size_t GetTargetRank(size_t j, size_t weight) {
        if (weight % 2 == 0) { // even
            return j * weight + (weight + 2 * (j % 2)) / 2;
        } else { // uneven
            return j * weight + (weight + 1) / 2;
        }
    }
};

int main() {
    using ValueType = double;
    auto p = 4;
    auto b = 10;
    auto k = 60;
    auto N_pow = 7;
    auto N_p = ((int) (pow(10, N_pow) / (p * b * k)) + 1) * b * k;
    auto N = N_p * p;
    std::vector<OnlineSampler<ValueType>> buffers(p, OnlineSampler<ValueType>(b, k));
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni;
    std::lognormal_distribution<double> log_norm(0.0, 1.0); // TODO higher peak
    // TODO try sorted

    std::vector<ValueType> sequence;

    // stream data into buffers
    for (int i = 0; i < N_p; i++) {
        for (int j = 0; j < p; j++) {
            auto element = log_norm(rng);
            auto has_capacity = buffers[j].Put(element);
            sequence.emplace_back(element);
            if (!has_capacity) {
                buffers[j].Collapse([](ValueType element) {}); // TODO test convergence
            }
        }
    }

    // collapse until final samples
    bool collapsible;
    std::vector<std::vector<ValueType>> samples(p, std::vector<ValueType>());
    for (int j = 0; j < p; j++) {
        do {
            samples[j].clear();
            collapsible = buffers[j].Collapse([](ValueType element) {});
            buffers[j].GetHighestWeightedSamples(samples[j]);
        } while (collapsible);
    }

    // merge parallel samples
    OnlineSampler<ValueType> result_buffers(p, k);
    for (int j = 0; j < p; j++) {
        for (int i = 0; i < k; i++) {
            result_buffers.Put(samples[j][i]);
        }
    }
    std::vector<ValueType> result_samples;
    result_buffers.Collapse([](ValueType element) {});
    result_buffers.GetHighestWeightedSamples(result_samples);

    // calculate error
    std::sort(sequence.begin(), sequence.end());
    float error = 0;
    for (int j = 0; j < k; j++) {
        int target_rank = (N / k) * (j + 1);
        int actual_rank = 0;
        while (sequence[actual_rank] < result_samples[j]) {
            actual_rank++;
        }
        error += pow(target_rank - actual_rank, 2);
    }
    error = std::sqrt(error / (k - 1)) / N;
    std::cout << "p: " << p << ", N: " << N << ", b: " << b << ", k: " << k << " | " << error << "\n";

    return 0;
}