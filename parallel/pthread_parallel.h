#ifndef pthread_parallel_h
#define pthread_parallel_h

#include "mylib.h"


void pthread_with_numa(GlobalVar* g);
void pthread_without_numa(GlobalVar* g);
void pthread_with_numa_aligned(GlobalVar* g);


void allocate_matrix_by_row_time_col(pInfo* pthreads, GlobalVar* g, void *(*f)(void *));
void evaluate_access_time(pInfo* pthreads, GlobalVar* g, void *(*f)(void *));
void evaluate_multiplication_time(pInfo* pthreads, GlobalVar* g, void *(*f)(void *));
#endif
