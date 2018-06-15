#include "openmp_parallel.h"
#include <omp.h>

using namespace std;

void openmp_with_numa(GlobalVar* g) {
  
  long i = 0;
  long j = 0;
  float** mx = g->mx;
  float* v = g->v;
  float* w = g->w;
  int rows = g->rows;
  int cols = g->cols;
  map<int, int> cpu_to_node = g->cpu_to_node;
  map<int, int> thread_to_cpu = g->thread_to_cpu;
  long* time_records = g->time_records;
  int thread_id;
  int sched_cpu;

#pragma omp parallel proc_bind(close) num_threads(g->n_threads) default(none) shared(mx, rows, cols, thread_to_cpu, cpu_to_node) private(i, j, sched_cpu, thread_id)
  {
    sched_cpu = sched_getcpu();
    thread_id = omp_get_thread_num();
    
    if (thread_to_cpu[thread_id] != sched_cpu) {
      printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
    }
    
#pragma omp for ordered schedule (static) 
    for (i = 0; i < rows; i++) {
      mx[i] = (float*)numa_alloc_onnode(sizeof(float) * cols, cpu_to_node[sched_cpu]);
      for (j = 0; j < cols; j++) {
	mx[i][j] = 0;
      }
    }
  }
  
  /**evaluate access time*/
#pragma omp parallel proc_bind(close) num_threads(g->n_threads) default(none) shared(time_records, mx, rows, cols, thread_to_cpu, cpu_to_node) private(i, j, sched_cpu, thread_id) 
  {
    sched_cpu = sched_getcpu();
    thread_id = omp_get_thread_num();
    if (thread_to_cpu[thread_id] != sched_cpu) {
      printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
    }
    
    auto start = tick();
#pragma omp for schedule (static)
    for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
	mx[i][j] = 3;
      }
    }
    auto end = tick();
    time_records[thread_id] = end - start;
  }
  
  printf("=== check access time records\n");
  check_time_records(time_records, g->n_threads);


  /**evaluate multiplication time*/
#pragma omp parallel proc_bind(close) num_threads(g->n_threads) default(none) shared(time_records, mx, v, w, rows, cols, thread_to_cpu, cpu_to_node) private(i, j, sched_cpu, thread_id)
  {
    
    sched_cpu = sched_getcpu();
    thread_id = omp_get_thread_num();
    if (thread_to_cpu[thread_id] != sched_cpu) {
      printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
    }

    auto start = tick();
#pragma omp for schedule (static)
    for (i = 0; i < rows; i++) {
      for (j = 0; j < cols; j++) {
	w[i] += (mx[i][j] * v[j]);
      }
    }
    auto end = tick();
    time_records[thread_id] = end - start;
  }

  printf("=== check multiplication time records\n");
  check_time_records(time_records, g->n_threads);
  //check_mx(g);
  //check_v(g);
  //check_w(g);
}



void openmp_without_numa(GlobalVar* g) {
  long i = 0;
  long j = 0;
  float** mx = g->mx;
  float* v = g->v;
  float* w = g->w;
  int rows = g->rows;
  int cols = g->cols;

  long* time_records = g->time_records;
  int thread_id;
  int sched_cpu;
 
#pragma omp parallel for num_threads(g->n_threads) default(none) shared(mx, rows, cols) private(i, j) 
  for (i = 0; i < rows; i++) {
    mx[i] = (float*)emalloc(sizeof(float) * cols);
    for(j = 0; j < cols; j++) {
      mx[i][j] = 0;
    }
    check_address_if_on_node(&mx[i][63], i / 8, i, 63);
  }
 
  /**evaluate access time*/
  auto start = tick();
#pragma omp parallel for num_threads(g->n_threads) default(none) shared(mx, rows, cols) private(i, j)
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      mx[i][j] = 3;
    }
  }
  auto end = tick();
  printf("===parallel access use: %lld\n", end - start);

  start = tick();
#pragma omp parallel for num_threads(g->n_threads) default(none) shared(mx, v, w, rows, cols) private(i, j)
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      w[i] += (mx[i][j] * v[j]);
    }
  }
  end = tick();
  printf("===parallel matrix multiplication use: %lld\n", end - start);
}
