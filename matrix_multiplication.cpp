//
// Created by zwpdbh on 07/05/2018.
//

#include <pthread.h>
#include <iostream>
#include <chrono>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <map>
#include <vector>
#include <numa.h>
#include <chrono>


using namespace std;

long ROWS;
long COLS;
int NUM_THREADS = 64;
int NUM_NODES = 8;

float** mx = nullptr;
float* v = nullptr;
float* w = nullptr;
long s = 0;
int with_numa = 1;
map<int, int> topology;
map<int, vector<int> > topology_inverse;

typedef struct pthreadInfo {
  int thread_id;
  long from;
  long job_size;
  int row_in_mx;
  int p;
  pthread_t thread;
} pInfo;

void *emalloc(size_t s);
void *remalloc(void *p, size_t s);
void* thr_alloc_fn(void* arg);
void* thr_access_fn(void* arg);
void* thr_multiply_fn(void* arg);
void check_mx();
void check_w();
void construct_barrier(pInfo* pthreads, int n_threads);
void initMap(map<int, int> &topology, map<int, vector<int> > &topology_inverse);

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("usage: ./bin/matrix_multiplication <rows> <cols> <with_numa>, exit...\n");
    exit(-1);
  }
  ROWS = atol(argv[1]);
  COLS = atol(argv[2]);
  with_numa = atoi(argv[3]);
  if (with_numa) {
    printf("WITH numa\n");
  } else {
    printf("WITHOUT numa configuration\n");
  }
  s = ROWS * COLS;
  printf("s = %ld\n", s);
  initMap(topology, topology_inverse);

  mx = (float**)emalloc(sizeof(float*) * NUM_NODES);
  pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * NUM_NODES);
  pthread_t* thread; // array of pthread_t, for catching each created thread
  int err = 0;


  /**Allocate m by n matrix with the shape NUM_NODES * some right size*/
  for (int i = 0; i < NUM_NODES; i++) {
    pthreads[i].thread_id = i;
    pthreads[i].row_in_mx = i;
    pthreads[i].job_size = (s/NUM_NODES);
    if (pthreads[i].thread_id == NUM_NODES - 1) {
      pthreads[i].job_size += (s % NUM_NODES);
    }
    err = pthread_create(&pthreads[i].thread, NULL, thr_alloc_fn, (void *)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit..\n");
      exit(-1);
    }
  }

  /**manually created barrier*/
  construct_barrier(pthreads, NUM_NODES);
  //  check_mx();

  auto started = std::chrono::high_resolution_clock::now();

  pthreads =(pInfo*)remalloc(pthreads, sizeof(*pthreads) * NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; i++) {
    pthreads[i].thread_id = i;
    pthreads[i].row_in_mx = i / (NUM_THREADS / NUM_NODES);
    pthreads[i].p = i  - pthreads[i].row_in_mx * (NUM_THREADS / NUM_NODES);
    pthreads[i].job_size = s / NUM_THREADS;
    pthreads[i].from = pthreads[i].p * pthreads[i].job_size;
    if ((pthreads[i].p + 1) % (NUM_THREADS / NUM_NODES) == 0) {
      pthreads[i].job_size += ((s / NUM_NODES) % (NUM_THREADS / NUM_NODES));
    }
    if (pthreads[i].thread_id == NUM_THREADS - 1) {
      pthreads[i].job_size +=  (s % NUM_NODES);
    }

    err = pthread_create(&pthreads[i].thread, NULL, thr_access_fn, (void *)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit...\n");
      exit(-1);
    }
  }

  construct_barrier(pthreads, NUM_THREADS);
  auto done = std::chrono::high_resolution_clock::now();
  std::cout << "From initialization to finished, use: " << std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count() << "ns" << endl;
  printf("\n");
  //  check_mx();


  /**do the multiplication with a vector
  v = (float*)emalloc(sizeof(float) * COLS);
  w = (float*)emalloc(sizeof(float) * ROWS);
  for (int i = 0; i < COLS; i++) {
    v[i] = 1;
  }
  for (int i = 0; i < ROWS; i++) {
    w[i] = 0;
  }

  // distribute multiplication task on different threads
  for (int i = 0; i < NUM_THREADS; i++) {
    if (ROWS / NUM_THREADS == 0) {
      pthreads[i].from = i * 1;
      pthreads[i].job_size = 1;
    } else {
      pthreads[i].from  = i * (ROWS / NUM_THREADS);
      pthreads[i].job_size = ROWS / NUM_THREADS;
    }

    if (i == NUM_THREADS - 1) {
      pthreads[i].job_size += (ROWS % NUM_THREADS);
    }
    err = pthread_create(&pthreads[i].thread, NULL, thr_multiply_fn, (void*)&pthreads[i]);
    if (err != 0) {
      printf("error during thread creation, exit...\n");
      exit(-1);
    }
  }


  construct_barrier(pthreads, NUM_THREADS);
  check_w();
  */

  for (int i = 0; i < NUM_NODES; i++) {
    //    free(mx[i]);
    if (i == NUM_NODES - 1) {
      numa_free(mx[i], (s / NUM_NODES + s % NUM_NODES) * sizeof(float));
    } else {
      numa_free(mx[i], (s / NUM_NODES) * sizeof(float));
    }
  }

  free(mx);
  free(pthreads);
  exit(0);
}

void *emalloc(size_t s) {
  void *result = malloc(s);
  if (result == nullptr) {
    fprintf(stderr, "memory allocation failed");
    exit(EXIT_FAILURE);
  }
  return result;
}

void *remalloc(void *p, size_t s) {
  void *result = realloc(p, s);
  if (result == NULL) {
    fprintf(stderr, "memory allocation failed");
    exit(EXIT_FAILURE);
  }
  return result;
}

// thread function to use libnuma to allocate memory on specific node
void* thr_alloc_fn(void* arg) {
  pInfo info = *(pInfo*)arg;

  mx[info.thread_id] = (float*)numa_alloc_onnode(info.job_size * sizeof(float), info.thread_id);
  for (int k = 0; k < info.job_size; k++) {
    mx[info.row_in_mx][k] = 0;
  }

  return (void *)0;
}


void* thr_access_fn(void* arg) {
  pid_t pid = getpid();
  int sched_cpu = sched_getcpu();
  int current_on_node = topology[sched_cpu];

  pInfo info = *(pInfo *)arg;
  int node_index = info.thread_id / (NUM_THREADS / NUM_NODES);
  if(with_numa && node_index != current_on_node) {
    //    printf("thread_id = %d, on cpu %d, on node %d, should be on node%d\n", (unsigned long)info.thread_id, sched_cpu, current_on_node, node_index);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    vector<int> cpus = topology_inverse[node_index];
    for (int i = 0; i < cpus.size(); i++) {
      CPU_SET(cpus[i], &cpuset);
    }
    pthread_t tid = pthread_self();
    pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
    sched_cpu = sched_getcpu();
    current_on_node = topology[sched_cpu];
    //    printf("thread_id = %d, on cpu %d, on node %d\n", (unsigned long)info.thread_id, sched_cpu, current_on_node);
  }

  for (long i = info.from; i < info.from + info.job_size; i++) {
    mx[info.row_in_mx][i] = 3.0;
  }
  return (void *)0;
}

void* thr_multiply_fn(void* arg) {
  //    eachInfo info = *(eachInfo*)arg;
  pInfo info = *(pInfo*)arg;
  for (int r = info.from; r < info.from  + info.job_size; ++r) {
    // get corresponding mx(i, j)
    for (int i = 0; i < COLS; i++) {
      long l = r * COLS + i;
      long row_in_mx = l / (s / NUM_NODES) - 1;
      if (row_in_mx < 0) {
	row_in_mx = 0;
      }
      long col_in_mx = l % (s / NUM_NODES);
      w[r] += (*(*(mx + row_in_mx) + col_in_mx) * v[i]);
    }
  }

  return (void *)0;
}

void check_mx() {
  // check the content of mx
  long size_for_each_node = s / NUM_NODES;

  for(int i = 0; i < NUM_NODES; i++) {
    if (i == NUM_NODES -1) {
      size_for_each_node += s % NUM_NODES;
    }
    for (int j = 0; j < size_for_each_node; j++) {
      printf("%.f ", *(*(mx + i) + j));
    }
    printf("\n");
  }
}

void check_w() {
  printf("w = \n");
  for (long i = 0; i < ROWS; i++) {
    printf("%.f ", w[i]);
  }
  printf("\n");
}

void construct_barrier(pInfo* pthreads, int n_threads) {
  int err;
  void* status;
  for (int i = 0; i < n_threads; i++) {
    err = pthread_join(pthreads[i].thread, &status);
    if (err) {
      printf("error, return code from pthread_join() is %d\n", *(int*)status);
    }
  }
}


void initMap(map<int, int> &topology, map<int, vector<int> > &topology_inverse) {

  vector<int> index_set = {0, 32, 2, 34, 3, 35, 1, 33};
  for (int node = 0 ; node < index_set.size(); node++) {
    vector<int> cpus_on_node;
    for (int cpu = index_set[node]; cpu < index_set[node] + 32; cpu+=4) {
      //      printf("make pair: %d <=> %d\n", node, cpu);
      topology.insert(make_pair(cpu, node));
      cpus_on_node.push_back(cpu);
    }
    topology_inverse.insert(make_pair(node, cpus_on_node));
  }
}
