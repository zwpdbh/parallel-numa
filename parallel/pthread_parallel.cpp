#include "pthread_parallel.h"

/**currently it is as same as the pthread_with_numa, 
 *maybe later change this part to test extra alignment*/
void pthread_with_numa_aligned(GlobalVar* g) {
  pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * g->n_threads);
  allocate_matrix_by_row_time_col(pthreads, g, alloc_fn);
  evaluate_access_time(pthreads, g, access_fn);
  evaluate_multiplication_time(pthreads, g, multiply_fn);
  int cpu_num = sched_getcpu();
  if(cpu_num) {
    printf("main thread is not fixed on cpu0\n");
  }
  free(pthreads);
}

void pthread_with_numa(GlobalVar* g) {

  pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * g->n_threads);

  allocate_matrix_by_row_time_col(pthreads, g, alloc_fn);
  evaluate_access_time(pthreads, g, access_fn);
  evaluate_multiplication_time(pthreads, g, multiply_fn);

  int cpu_num = sched_getcpu();
  if(cpu_num) {
    printf("main thread is not fixed on cpu0\n");
  }

  free(pthreads);
}


void pthread_without_numa(GlobalVar* g) {
  
  pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * g->n_threads);

  allocate_matrix_by_row_time_col(pthreads, g, alloc_no_numa_fn);
  evaluate_access_time(pthreads, g, access_no_numa_fn);
  evaluate_multiplication_time(pthreads, g, multiply_no_numa_fn);

  free(pthreads);
}



void allocate_matrix_by_row_time_col(pInfo* pthreads, GlobalVar* g, void *(*f)(void *)) {
  int err = 0;
  printf("===Allocate mx with %ld * %ld, WITH NUMA:\n", g->rows, g->cols);

  for (int i = 0; i < g->n_threads; i++) {
    pthreads[i].g = g;
    pthreads[i].thread_id = i;
    pthreads[i].job_size = g->rows / g->n_threads;
    pthreads[i].from = i * (g->rows / g->n_threads);

    if (pthreads[i].thread_id == g->n_threads -1) {
      pthreads[i].job_size += (g->rows % g->n_threads);
    }

    err = pthread_create(&pthreads[i].thread, NULL, f, (void *)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit..\n");
      exit(-1);
    }
  }
  construct_barrier(pthreads, g->n_threads);
}

void evaluate_access_time(pInfo* pthreads, GlobalVar* g, void* f(void*)) {
  int err = 0;
  /**evaluation access time*/
  for (int i = 1; i < g->n_threads; i++) {
    pthreads[i].g = g;
    pthreads[i].thread_id = i;
    pthreads[i].from = i * (g->rows / g->n_threads);
    pthreads[i].job_size = g->rows / g->n_threads;
    if (pthreads[i].thread_id == g->n_threads - 1) {
      pthreads[i].job_size += (g->rows % g->n_threads);
    }
    err = pthread_create(&pthreads[i].thread, NULL, f, (void*)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit...\n");
      exit(-1);
    }
  }

  pthreads[0].g = g;
  pthreads[0].thread_id = 0;
  pthreads[0].from = 0;
  pthreads[0].job_size = g->rows / g->n_threads;

  fix_current_thread_to_cpu(0);
  pthread_barrier_wait(&g->b);
  do_access(&pthreads[0]);
  pthread_barrier_wait(&g->b);

  // construct_barrier(pthreads, g->n_threads - 1);
  printf("===check access time records:\n");
  check_time_records(g->time_records, g->n_threads);
}

void evaluate_multiplication_time(pInfo* pthreads, GlobalVar* g, void* f(void*)) {
  int err = 0;

  for (int i = 1; i < g->n_threads; i++) {
    pthreads[i].g = g;
    pthreads[i].thread_id = i;
    // for the case in which row < n_threads
    if (g->rows / g->n_threads == 0) {
      pthreads[i].from = i * 1;
      pthreads[i].job_size = 1;
    } else {
      pthreads[i].from = i * (g->rows / g->n_threads);
      pthreads[i].job_size = g->rows / g->n_threads;
    }
    if (i == g->n_threads) {
      pthreads[i].job_size += (g->rows % g->n_threads);
    }

    err = pthread_create(&pthreads[i].thread, NULL, f, (void*)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit...\n");
      exit(-1);
    }
  }

  pthreads[0].thread_id = 0;
  pthreads[0].g = g;
  pthreads[0].from = 0;
  pthreads[0].job_size = g->rows / g->n_threads;

  fix_current_thread_to_cpu(0);
  pthread_barrier_wait(&g->b);
  do_multiplication(&pthreads[0]);
  pthread_barrier_wait(&g->b);

  //  construct_barrier(pthreads, g->n_threads - 1);
  //  this function contains unknown problem

  printf("=== check multiplication time_records:\n");
  check_time_records(g->time_records, g->n_threads);
}


