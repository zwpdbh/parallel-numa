#include <omp.h>
#include <map>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <numa.h>
#include <sched.h>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <iostream>

using namespace std;

long ROWS;
long COLS;
int NUM_THREADS = 64;
int NUM_NODES = 8;
pthread_t main_thread;
int with_numa = 1;

float** mx = nullptr;
float* v = nullptr;
float* w = nullptr;
long s = 0;
long* time_records;
map<int, int> cpu_to_node;
map<int, int> thread_to_cpu;
map<int, vector<int> > node_to_cpus;


void *emalloc(size_t s);
void *remalloc(void *p, size_t s);
void initMap(map<int, int> &cpu_to_node, map<int, vector<int> > &node_to_cpus, map<int, int> &thread_to_cpu);
void check_w();
void check_time_records(long* time_records, int n);


inline uint64_t tick() {
  uint32_t tmp[2];
  __asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
  return (((uint64_t) tmp[0]) << 32) | tmp[1];
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("usage: ./bin/xxx <rows> <cols> <1 or 0>, exit...\n");
    exit(-1);
  }
  
  if(numa_available() == -1) {
    printf("no libnuma support\n");
  } else  {
    numa_set_strict(1);    
  }

  printf("OMP_PLACES : %s\n", getenv("OMP_PLACES"));
  ROWS = atol(argv[1]);
  COLS = atol(argv[2]);
  with_numa = atoi(argv[3]);

  initMap(cpu_to_node, node_to_cpus, thread_to_cpu);
  
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  main_thread = pthread_self();
  pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &cpuset);
  printf("fix main_thread: %lu on cpu0\n", main_thread);

  v = (float*)emalloc(sizeof(float) * COLS);
  w = (float*)emalloc(sizeof(float) * ROWS);
  mx = (float**)emalloc(sizeof(float*) * ROWS);
  time_records = (long*)emalloc(sizeof(long) * NUM_THREADS);

  long i = 0;
  long j = 0;

  for (j = 0; j < COLS; j++) {
    v[j] = 1;
  }
  for (i = 0; i < ROWS; i++) {
    w[i] = 0;
  }
  //  check_w();

  auto start = tick();
  auto end = tick();

  if (with_numa) {
    auto started = std::chrono::high_resolution_clock::now();
    printf("===WITH NUMA:\n");
    /**allocate mx among different nodes*/
#pragma omp parallel proc_bind(close) num_threads(NUM_THREADS) default(none) shared(mx, cpu_to_node, ROWS, COLS, thread_to_cpu) private(i, j)
    {
      int sched_cpu = sched_getcpu();
      int thread_id = omp_get_thread_num();

      if (thread_to_cpu[thread_id] != sched_cpu) {
	printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
      }

#pragma omp for ordered schedule (static)
      for(i = 0; i < ROWS; i++) {
	mx[i] = (float*)numa_alloc_onnode(sizeof(float) * COLS, cpu_to_node[sched_cpu]);
      }
    }

    /**measure access time*/
    auto acc_start = tick();
    int thread_id;
#pragma omp parallel proc_bind(close) num_threads(NUM_THREADS) default(none) shared(time_records, mx, cpu_to_node, ROWS, COLS, thread_to_cpu) private(i,j, start, end)
    {
      int sched_cpu = sched_getcpu();
      int thread_id = omp_get_thread_num();

      if (thread_to_cpu[thread_id] != sched_cpu) {
	printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
      }

      start = tick();
#pragma omp for schedule (static)
      for (i = 0; i < ROWS; i++) {
	for (j = 0; j < COLS; j++) {
	  mx[i][j] = 3;
	}
      }
      end = tick();
      time_records[thread_id] = end - start;
    }
    auto acc_end= tick();
    check_time_records(time_records, NUM_THREADS);
    printf("acess uses: %ld\n", acc_end - acc_start);
    //    check_mx();
    /**measure multiplication time*/

    auto mul_start = tick();
    printf("during multiplication:\n");
#pragma omp parallel proc_bind(close) num_threads(NUM_THREADS) default(none) shared(mx, v, w, cpu_to_node, ROWS, COLS, thread_to_cpu, time_records) private(i, j, start, end, thread_id)
    {
      int sched_cpu = sched_getcpu();
      thread_id = omp_get_thread_num();
      
      if (thread_to_cpu[thread_id] != sched_cpu) {
	printf("thread_%d should execute on cpu_%d, but it is running on %d\n", thread_id, thread_to_cpu[thread_id], sched_cpu);
      }

      start = tick();
#pragma omp for schedule (static)
      for (i = 0; i < ROWS; i++) {
	for(j = 0; j < COLS; j++) {
	  w[i] += (mx[i][j] +  v[j]);
	  mx[i][j] = 4;
	}
      }
      end = tick();
      time_records[thread_id] = end - start;
    }
    if (sched_getcpu()) {
      printf("main thread is not fixed on CPU_01\n");
    }
    auto mul_end = tick();
    check_time_records(time_records, NUM_THREADS);
    printf("parallel matrix multiplication use: %lld\n", mul_end - mul_start);

    //    check_mx();
    //    check_w();

    for(i = 0; i < ROWS; i++) {
      numa_free(mx[i], sizeof(float) * COLS);
    }
    free(mx);
    free(v);
    free(w);
    free(time_records);
    auto done = std::chrono::high_resolution_clock::now();
    cout << "total execution time: " << std::chrono::duration_cast<std::chrono::duration<double> >(done - started).count() << " seconds" << endl;
    exit(0);
  } else {

    auto started = std::chrono::high_resolution_clock::now();
    printf("===WITHOUT NUMA:\n");
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(mx, ROWS, COLS) private(i, j)
    for (i = 0; i < ROWS; i++) {
      mx[i] = (float*)emalloc(sizeof(float) * COLS);
    }

    auto start = tick();
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(mx, ROWS, COLS) private(i, j)
    for (i = 0; i < ROWS; i++) {
      for (j = 0; j < COLS; j++) {
	mx[i][j] = 3;
      }
    }
    auto end = tick();
    printf("parallel access use: %lld\n", end - start);

    start = tick();
#pragma omp parallel for num_threads(NUM_THREADS) default(none) shared(mx, v, w, ROWS, COLS) private(i, j)
    for (i = 0; i < ROWS; i++) {
      for (j = 0; j < COLS; j++) {
	w[i] += (mx[i][j] * v[j]);
      }
    }
    end = tick();
    printf("parallel matrix multiplication use: %lld\n", end - start);

    for(i = 0; i < ROWS; i++) {
      free(mx[i]);
    }
    free(mx);
    free(v);
    free(w);
    free(time_records);
    auto done = std::chrono::high_resolution_clock::now();
    cout << "total execution time: " << std::chrono::duration_cast<std::chrono::duration<double> >(done - started).count() << " seconds" << endl;
    exit(0);
  }
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

void initMap(map<int, int> &topology, map<int, vector<int> > &topology_inverse, map<int, int> &distribute_scheme) {
  int thread_id = 0;
  vector<int> index_set = {0, 32, 2, 34, 3, 35, 1, 33};
  for (int node = 0 ; node < index_set.size(); node++) {
    vector<int> cpus_on_node;
    for (int cpu = index_set[node]; cpu < index_set[node] + 32; cpu+=4) {
      //      printf("make pair: %d <=> %d\n", node, cpu);
      cpu_to_node.insert(make_pair(cpu, node));
      cpus_on_node.push_back(cpu);
      thread_to_cpu.insert(make_pair(thread_id, cpu));
      thread_id += 1;
    }
    node_to_cpus.insert(make_pair(node, cpus_on_node));
  }
}

void check_w() {

  for (long k = 0; k < ROWS; k++) {
    printf("%.f ", w[k]);
  }

}


void check_time_records(long* time_records, int n) {
  long max = 0;
  int index = 0;
  for (int i = 0; i < n; i++) {
    printf("thread_%d takes: %ld\n", i, time_records[i]);
    if(time_records[i] > max) {
      index = i;
      max = time_records[i];
    }
  }
  printf("thread_%d takes longest, uses: %ld\n", index, max);
}
