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

template <typename ValueType>
class Buffer {
public:
    size_t level;
    size_t weight;
    std::vector<ValueType> buffer;
    size_t current_index;

    Buffer(size_t k) {
        buffer = std::vector<ValueType>(k);
        Reset();
    }

    bool Put(ValueType value) {
        assert(HasCapacity());
        buffer[current_index++] = value;
        return HasCapacity();
    }

    bool HasCapacity() {
        return current_index < buffer.size();
    }

    bool IsEmpty() {
        return current_index == 0;
    }

    void Reset() {
        level = -1;
        weight = 0;
        current_index = 0;
    }
};

template <
        typename ValueType>
class Buffers {
public:
    Buffers(size_t b, size_t k) {
        k_ = k;
        buffers = std::vector<Buffer<ValueType>>(b, Buffer<ValueType>(k));
        current_buffer = -1;
        empty_buffers = buffers.size();
        minimum_level = 0;
    }

    bool Put(ValueType value) {
        if (current_buffer == -1 || !buffers[current_buffer].HasCapacity()) {
            New();
        }
        return buffers[current_buffer].Put(value) || empty_buffers > 0;
    }

    bool Collapse(std::vector<ValueType>& out_elements, std::vector<ValueType>& splitters) {
        std::vector<Buffer<ValueType> *> working_buffers;
        std::vector<size_t> indices;
        size_t weight_sum = 0;
        for (int i = 0; i < buffers.size(); i++) {
            if (buffers[i].level == minimum_level && !buffers[i].IsEmpty()) {
                std::sort(buffers[i].buffer.begin(), buffers[i].buffer.end());
                working_buffers.emplace_back(&buffers[i]);
                indices.emplace_back(i);
                weight_sum += buffers[i].weight;
            }
        }
        Buffer<ValueType> target_buffer(k_);
        target_buffer.level = ++minimum_level;
        target_buffer.weight = weight_sum;
        size_t total_index = 0;
        std::vector<size_t> positions(working_buffers.size(), 0);
        for (int j = 0; j < k_; j++) {
            size_t target_rank;
            if (target_buffer.weight % 2 == 0) { // even
                target_rank = j * target_buffer.weight + (target_buffer.weight + 2 * (j % 2)) / 2;
            } else { // uneven
                target_rank = j * target_buffer.weight + (target_buffer.weight + 1) / 2;
            }
            // TODO switch to tournament tree
            while (total_index < target_rank) {
                ValueType minimum = working_buffers[0]->buffer[positions[0]];
                size_t minimum_index = 0;
                for (int i = 1; i < working_buffers.size(); i++) {
                    ValueType value = working_buffers[i]->buffer[positions[i]];
                    if (value < minimum) {
                        minimum = value;
                        minimum_index = i;
                    }
                }
                total_index += working_buffers[minimum_index]->weight;
                positions[minimum_index]++;
                out_elements.emplace_back(minimum);
            }
            auto splitter = out_elements.back();
            splitters.emplace_back(splitter);
            out_elements.pop_back();
            target_buffer.Put(splitter);
        }
        buffers[indices[0]] = target_buffer;
        for (int i = 1; i < working_buffers.size(); i++) {
            working_buffers[i]->Reset();
        }
        current_buffer = -1;
        empty_buffers += working_buffers.size() - 1;
        return buffers.size() - empty_buffers > 1;
    }

private:
    size_t k_;

    std::vector<Buffer<ValueType>> buffers;
    size_t current_buffer;
    size_t empty_buffers;
    size_t minimum_level;

    void New() {
        for (int i = 0; i < buffers.size(); i++) {
            if (buffers[i].IsEmpty()) {
                current_buffer = i;
                if (empty_buffers > 1) {
                    buffers[current_buffer].level = 0;
                    minimum_level = 0;
                } else {
                    buffers[current_buffer].level = minimum_level;
                }
                buffers[current_buffer].weight = 1;
                empty_buffers -= 1;
                break;
            }
        }
    }
};

int main() {
    auto b = 5;
    auto k = 100;
    Buffers<int> buffers(b, k);
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni;

    for (int i = 0; i < 10 * b * k; i++) {
        auto has_capacity = buffers.Put(uni(rng));
        if (!has_capacity) {
            std::vector<int> discarded_elements;
            std::vector<int> splitters;
            buffers.Collapse(discarded_elements, splitters);
        }
    }
    bool collapsible;
    do {
        std::vector<int> discarded_elements;
        std::vector<int> splitters;
        collapsible = buffers.Collapse(discarded_elements, splitters);
    } while(collapsible);

    return 0;
}