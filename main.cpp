#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>
#include <functional>


#define DTYPE int
const size_t num_data (1<<24);
auto ascent = [] (DTYPE a, DTYPE b) {return a<b;};
auto descent = [] (DTYPE a, DTYPE b) {return a>b;};

// For debugging
void print(std::vector<DTYPE>& data);
void test_result(std::vector<DTYPE>& data, std::vector<DTYPE>& gt);

// Bitonic sorting algorithm
void bitonic_sort(std::vector<DTYPE>& data);
void bitonic_generate(std::vector<DTYPE>& data, size_t start, size_t end);
template<typename F> void bitonic_merge(std::vector<DTYPE>& data, size_t start, size_t end, F comparer);
template<typename F> void bitonic_split(std::vector<DTYPE>& data, size_t start, size_t end, F comparer);




int main(void) {
    srand(time(NULL));
    std::vector<DTYPE> data(num_data);
    std::generate(data.begin(), data.end(), [](){return (std::rand()%num_data);});
    std::vector<DTYPE> gt(data);
    std::sort(gt.begin(), gt.end());

    std::cout << "Parallel Bitonic Sorting start with data[0:" << data.size()-1 << "]\n";
    timeval time_start, time_end;
    gettimeofday(&time_start, NULL);
    bitonic_sort(data);
    gettimeofday(&time_end, NULL);
    float time = (time_end.tv_sec - time_start.tv_sec) + ((time_end.tv_usec-time_start.tv_usec)*1e-6);
    std::cout << "--- Elapsed time: " << time << " s"<< std::endl;

    test_result(data, gt);
    return 0;
}

void bitonic_sort(std::vector<DTYPE>& data) {
    bitonic_generate(data, 0, data.size()-1);
    bitonic_merge(data, 0, data.size()-1, ascent);
}

void bitonic_generate(std::vector<DTYPE>& data, size_t start, size_t end) {
    
    if (start == end)
        return;

    size_t mid = (start+end)/2+1;
    bitonic_generate(data, start, mid-1);
    bitonic_generate(data, mid, end);
    bitonic_merge(data, start, mid-1, ascent);
    bitonic_merge(data, mid, end, descent);
}

template<typename F> void bitonic_merge(std::vector<DTYPE>& data, size_t start, size_t end, F comparer) {
    if (start == end)
        return;
    size_t mid = (start+end)/2+1;
    bitonic_split(data, start, end, comparer);
    bitonic_merge(data, start, mid-1, comparer);
    bitonic_merge(data, mid, end, comparer);
}   

template<typename F> void bitonic_split(std::vector<DTYPE>& data, size_t start, size_t end, F comparer) {
    
    size_t mid = (start+end)/2+1;

    for (int i=start; i<mid; i++) {
        DTYPE a = data[i], b = data[i+(mid-start)];
        data[i] = comparer(a, b) ? a : b;
        data[i+(mid-start)] = !comparer(a, b) ? a : b;
    }
}


void test_result(std::vector<DTYPE>& data, std::vector<DTYPE>& gt) {

    bool result = true;
    for (size_t i=0; i<data.size(); i++) {
        if (data[i] != gt[i]) {
            result = false;
            break;
        }
    }

    if (result == true) {
        std::cout << "--- Ascent sorting is correct!!" << std::endl;
    } else {
        std::cout << "[[ERROR]] Ascent sorting fails!!" << std::endl;
    } 

};

void print(std::vector<DTYPE>& data) {
    for (auto i=data.begin(); i<data.end(); i++) {
        std::cout << *i << " ";
    }
    std::cout << std::endl;
}