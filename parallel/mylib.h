#ifndef mylib_h
#define mylib_h

#include <omp.h>
#include <map>
#include <pthread.h>
#include <sys/types.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <unistd.h>

class MyMatrix;

inline uint64_t tick() {
  uint32_t tmp[2];
  __asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
  return (((uint64_t) tmp[0]) << 32) | tmp[1];
}

struct GlobalVar;

typedef struct pthreadInfo {
  int thread_id;
  long from;
  long job_size;
  long row_in_mx;
  int p;
  pthread_t thread;
  GlobalVar* g;
} pInfo;


struct GlobalVar {
  GlobalVar(long rows, long cols, int n_threads, int n_nodes, int option);
  ~GlobalVar();

  long rows;
  long cols;
  int n_threads;
  int n_nodes;
  long* time_records;
  int option;

  float** mx;
  float** unAlignedMx;
  float* v;
  float* unAlignedV;
  float* w;
  float* unAlignedW;

  std::map<int, int> cpu_to_node;
  std::map<int, int> thread_to_cpu;
  std::map<int, std::vector<int> > node_to_cpus;
  pthread_barrier_t b;
  MyMatrix* mxPtr;
};

typedef struct aligned_ptr_struct {
  void* actual_allocated_ptr;
  void* actual_used_ptr;
} *aligned_ptr;


extern void *emalloc(size_t s);
extern void *remalloc(void *p, size_t s);
extern void fix_current_thread_to_cpu(int cpu_id);
extern void migrate_to_node(int node_id, std::map<int, std::vector<int> > topology_inverse);
extern void check_time_records(long* time_records, int n);
extern void construct_barrier(pInfo* pthreads, int n_threads);


extern void* access_fn(void* arg);
extern void* alloc_fn(void* arg);
extern void* multiply_fn(void* arg);

extern void* access_no_numa_fn(void* arg);
extern void* alloc_no_numa_fn(void* arg);
extern void* multiply_no_numa_fn(void* arg);

extern void do_access(pInfo* p);
extern void do_multiplication(pInfo* p);

extern void check_mx(GlobalVar* g);
extern void check_v(GlobalVar* g);
extern void check_w(GlobalVar* g);

extern void check_aligned(long address, int align_size, const char* s);
extern void check_address_if_on_node(void* address, int node_id, long i, long j);
#endif
