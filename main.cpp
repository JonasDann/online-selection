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

template <typename ValueType>
class Buffer {
public:
    size_t weight;
    std::vector<ValueType> elements;
    size_t current_index;
    bool sorted;

    explicit Buffer(size_t k) {
        elements = std::vector<ValueType>(k);
        Reset();
    }

    bool Put(ValueType value) {
        assert(HasCapacity());
        elements[current_index++] = value;
        return HasCapacity();
    }

    bool HasCapacity() {
        return current_index < elements.size();
    }

    bool IsEmpty() {
        return current_index == 0;
    }

    void Reset() {
        weight = 0;
        current_index = 0;
        sorted = false;
    }
};

template <
        typename ValueType>
class Buffers {
public:
    Buffers(size_t b, size_t k) : b_(b), k_(k), current_buffer(-1), empty_buffers(b),
            minimum_level(0) {
        buffer_pool = std::vector<Buffer<ValueType>>(b + 1, Buffer<ValueType>(k));
        leveled_buffers.emplace_back(std::vector<Buffer<ValueType> *>());
    }

    bool Put(ValueType value) {
        if (current_buffer == -1 || !GetCurrentBuffer()->HasCapacity()) {
            New();
        }
        return GetCurrentBuffer()->Put(value) || empty_buffers > 0;
    }

    bool Collapse(std::vector<ValueType>& out_elements) {
        auto level = leveled_buffers[minimum_level];
        size_t weight_sum = 0;
        for (int i = 0; i < level.size(); i++) {
            auto buffer = level[i];
            if (!buffer->sorted) {
                std::sort(buffer->elements.begin(), buffer->elements.end());
                buffer->sorted = true;
            }
            weight_sum += buffer->weight;
        }
        auto target_buffer = &buffer_pool[GetEmptyBufferIndex()];
        target_buffer->weight = weight_sum;
        target_buffer->sorted = true;
        size_t total_index = 0;
        std::vector<size_t> positions(level.size(), 0);
        for (int j = 0; j < k_; j++) {
            size_t target_rank = GetTargetRank(j, target_buffer->weight);
            // TODO switch to tournament tree
            while (total_index < target_rank) {
                size_t minimum_index = 0;
                while (positions[minimum_index] >= level[minimum_index]->elements.size()) { // empty buffers
                    minimum_index++;
                }
                ValueType minimum = level[minimum_index]->elements[positions[minimum_index]];
                for (int i = minimum_index + 1; i < level.size(); i++) {
                    if (positions[i] < level[i]->elements.size()) { // not empty
                        ValueType value = level[i]->elements[positions[i]];
                        if (value < minimum) {
                            minimum = value;
                            minimum_index = i;
                        }
                    }
                }
                total_index += level[minimum_index]->weight;
                positions[minimum_index]++;
                out_elements.emplace_back(minimum);
            }
            out_elements.pop_back();
            target_buffer->Put(out_elements.back());
        }

        for (int i = 0; i < level.size(); i++) {
            level[i]->Reset();
        }
        empty_buffers += level.size() - 1;
        leveled_buffers[minimum_level].clear();
        current_buffer = -1;

        minimum_level++;
        if (minimum_level >= leveled_buffers.size()) {
            leveled_buffers.emplace_back(std::vector<Buffer<ValueType> *>());
        }
        assert(minimum_level < leveled_buffers.size());
        leveled_buffers[minimum_level].emplace_back(target_buffer);

        return b_ - empty_buffers > 1;
    }

    // TODO pseudo concat to use all knowledge in the buffers
    size_t GetSamples(std::vector<ValueType>& out_samples) {
        Buffer<ValueType> *max_weighted_buffer = &buffer_pool[0];
        for (int i = 1; i < buffer_pool.size(); i++) {
            if (buffer_pool[i].weight > max_weighted_buffer->weight) {
                max_weighted_buffer = &buffer_pool[i];
            }
        }
        out_samples = max_weighted_buffer->elements;
        return max_weighted_buffer->weight;
    }

private:
    size_t b_;
    size_t k_;

    std::vector<Buffer<ValueType>> buffer_pool;
    using LeveledBuffers = std::vector<std::vector<Buffer<ValueType> *>>;
    LeveledBuffers leveled_buffers;
    size_t current_buffer;
    size_t empty_buffers;
    size_t minimum_level;

    size_t GetEmptyBufferIndex() {
        for (int i = 0; i < buffer_pool.size(); i++) {
            if (buffer_pool[i].IsEmpty()) {
                return i;
            }
        }
        return -1;
    }

    void New() {
        current_buffer = GetEmptyBufferIndex();
        if (empty_buffers > 1) {
            leveled_buffers[0].emplace_back(GetCurrentBuffer());
            minimum_level = 0;
        } else {
            leveled_buffers[minimum_level].emplace_back(GetCurrentBuffer());
        }
        GetCurrentBuffer()->weight = 1;
        empty_buffers--;
    }

    size_t GetTargetRank(size_t j, size_t weight) {
        if (weight % 2 == 0) { // even
            return j * weight + (weight + 2 * (j % 2)) / 2;
        } else { // uneven
            return j * weight + (weight + 1) / 2;
        }
    }

    Buffer<ValueType> *GetCurrentBuffer() {
        return &buffer_pool[current_buffer];
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

    std::vector<Buffers<ValueType>> buffers(p, Buffers<ValueType>(b, k));
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
                std::vector<ValueType> discarded_elements;
                buffers[j].Collapse(discarded_elements);
                if (j == 0) {
                    std::vector<ValueType> samples;
                    auto sample_weight = buffers[j].GetSamples(samples);
                    print_convergence<ValueType>(i, sequence, last_sample_weight, sample_weight, samples);
                }
            }
        }

        // collapse until splitters
        bool collapsible;
        do {
            std::vector<ValueType> discarded_elements;
            collapsible = buffers[j].Collapse(discarded_elements);
            if (j == 0) {
                std::vector<ValueType> samples;
                auto sample_weight = buffers[j].GetSamples(samples);
                print_convergence<ValueType>(N_p, sequence, last_sample_weight, sample_weight, samples);
            }
        } while (collapsible);
        global_sequence.insert(global_sequence.end(), sequence.begin(), sequence.end());
    }
    std::cout << "\n";

    // merge parallel splitters
    Buffers<ValueType> global_buffers(p, k);
    for (int j = 0; j < p; j++) {
        std::vector<ValueType> samples;
        buffers[j].GetSamples(samples);
        for (int i = 0; i < k; i++) {
            global_buffers.Put(samples[i]);
        }
    }
    std::vector<ValueType> global_samples;
    if (p > 1) {
        std::vector<ValueType> discarded_elements;
        global_buffers.Collapse(discarded_elements);
    }
    global_buffers.GetSamples(global_samples);

    auto error = calculate_error<ValueType>(global_sequence, global_samples);
    std::cout << "final error: " << error << "\n\n";
}

int main() {
    std::uniform_int_distribution<int> uni;
    std::lognormal_distribution<double> log_norm(0.0, 0.1);

    online_sampling<double, std::lognormal_distribution<double>>(1, 10, 60, 7, log_norm, true);

    return 0;
}