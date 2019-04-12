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
        level_counters_.emplace_back(0);
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
        } else {
            level_counters_[minimum_level_]++;
        }
        assert(minimum_level_ < level_counters_.size());

        return b_ - empty_buffers_ > 1;
    }

    // TODO pseudo concat to use all knowledge in the buffers?
    size_t GetSamples(std::vector<ValueType> &out_samples) {
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
        buffers_[current_buffer_].sorted_ = false;
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

template <typename ValueType>
double calculate_error(std::vector<ValueType> sequence, std::vector<ValueType>& samples) {
    size_t N = sequence.size();
    size_t k = samples.size();
    std::sort(sequence.begin(), sequence.end());

    float error = 0;
    for (int j = 0; j < k; j++) {
        int target_rank = (N / k) * (j + 1);
        int actual_rank = 0;
        while (sequence[actual_rank] < samples[j]) {
            actual_rank++;
        }
        error += pow(target_rank - actual_rank, 2);
    }
    return std::sqrt(error / (k - 1)) / N;
}

template <typename ValueType>
void print_convergence(size_t i, std::vector<ValueType>& sequence, size_t& last_sample_weight, size_t sample_weight, std::vector<ValueType>& samples) {
    if (sample_weight > last_sample_weight) {
        auto error = calculate_error<ValueType>(sequence, samples);
        std::cout << "current error(" << i << ", "
                  << sample_weight << "): " << error << "\n";
    }
    last_sample_weight = sample_weight;
}

template <typename ValueType, typename Distribution>
void online_sampling(size_t p, size_t b, size_t k, size_t N_pow,
                     Distribution distribution, bool sorted) {
    auto N_p = ((int) (pow(10, N_pow) / (p * b * k)) + 1) * b * k;
    auto N = N_p * p;

    std::cout << "# Online Sampling" << "\n";
    std::cout << "p: " << p << ", b: " << b << ", k: " << k << ", N: " << N << ", sorted: " << sorted << "\n\n";

    std::vector<OnlineSampler<ValueType>> buffers(p, OnlineSampler<ValueType>(b, k));
    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<ValueType> global_sequence;
    global_sequence.reserve(p * N_p);
    for (int j = 0; j < p; j++) {
        // generate input sequence
        std::vector<ValueType> sequence;
        sequence.reserve(N_p);
        for (int i = 0; i < N_p; i++) {
            sequence.emplace_back(distribution(rng));
        }

        if (sorted) {
            std::sort(sequence.begin(), sequence.end());
        }

        // stream data into buffers
        size_t last_sample_weight = 0;
        for (int i = 0; i < N_p; i++) {
            auto has_capacity = buffers[j].Put(sequence[i]);
            if (!has_capacity) {
                buffers[j].Collapse([](ValueType element) {});
                if (j == 0) {
                    std::vector<ValueType> samples;
                    auto sample_weight = buffers[j].GetSamples(samples);
                    print_convergence<ValueType>(i, sequence, last_sample_weight, sample_weight, samples);
                }
            }
        }

        // collapse until final samples
        bool collapsible;
        do {
            collapsible = buffers[j].Collapse([](ValueType element) {});
            if (j == 0) {
                std::vector<ValueType> samples;
                auto sample_weight = buffers[j].GetSamples(samples);
                print_convergence<ValueType>(N_p, sequence, last_sample_weight, sample_weight, samples);
            }
        } while (collapsible);
        global_sequence.insert(global_sequence.end(), sequence.begin(), sequence.end());
    }
    std::cout << "\n";

    // merge parallel samples
    OnlineSampler<ValueType> global_buffers(p, k);
    for (int j = 0; j < p; j++) {
        std::vector<ValueType> samples;
        buffers[j].GetSamples(samples);
        for (int i = 0; i < k; i++) {
            global_buffers.Put(samples[i]);
        }
    }
    std::vector<ValueType> global_samples;
    if (p > 1) {
        global_buffers.Collapse([](ValueType element) {});
    }
    global_buffers.GetSamples(global_samples);

    auto error = calculate_error<ValueType>(global_sequence, global_samples);
    std::cout << "final error: " << error << "\n\n";
}

int main() {
    std::uniform_int_distribution<int> uni;
    std::lognormal_distribution<double> log_norm(0.0, 0.1);

    online_sampling<int, std::uniform_int_distribution<int>>(1, 10, 60, 7, uni, true);

    return 0;
}