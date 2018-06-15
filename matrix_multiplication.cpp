//
// Created by zwpdbh on 07/05/2018.
//

#include <pthread.h>
#include <iostream>
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
long* time_records;
int with_numa = 1;
map<int, int> topology;
map<int, int> distribute_scheme;
map<int, vector<int> > topology_inverse;

pthread_t main_thread;
pthread_barrier_t b;

inline uint64_t tick() {
  uint32_t tmp[2];
  __asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
  return (((uint64_t) tmp[0]) << 32) | tmp[1];
}

typedef struct pthreadInfo {
  int thread_id;
  long from;
  long job_size;
  long row_in_mx;
  int p;
  pthread_t thread;
} pInfo;

void *emalloc(size_t s);
void *remalloc(void *p, size_t s);
void* thr_alloc_fn(void* arg);
void* thr_access_fn(void* arg);
void* thr_multiply_fn(void* arg);
void* normal_alloc_fn(void* arg);
void* normal_access_fn(void* arg);
void* normal_multiply_fn(void* arg);
void* alloc_fn(void* arg);
void* access_fn(void* arg);
void* multiply_fn(void* arg);
void check_mx();
void check_w();
void check_normal_mx();
void construct_barrier(pInfo* pthreads, int n_threads);
void initMap(map<int, int> &topology, map<int, vector<int> > &topology_inverse);
void migrate_to_node(int node_index);
void do_multiplication(int thread_id, long from, long job_size);
void do_access(int thread_id, long from, long job_size);
void fix_current_thread_to_cpu(int num);
void check_time_records(long* time_records, int n);


int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("usage: ./bin/matrix_multiplication <rows> <cols> <with_numa>, exit...\n");
    printf("choose between -1, 0, 1, 2\n");
    exit(-1);
  }
  
  if(numa_available() == -1) {
    printf("no libnuma support\n");
  } else  {
    numa_set_strict(1);    
  }

  ROWS = atol(argv[1]);
  COLS = atol(argv[2]);
  with_numa = atoi(argv[3]);

  s = ROWS * COLS;
  printf("s = %ld\n", s);
  initMap(topology, topology_inverse);

  /**make sure the main thread is executed on a fixed cpu*/
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(0, &cpuset);
  main_thread = pthread_self();
  pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &cpuset);
  printf("fix main_thread: %lu on cpu0\n", main_thread);
  
  if (with_numa == -1) {
    auto started = std::chrono::high_resolution_clock::now();
    printf("===Allocate mx with %ld * %ld, WITHOUT NUMA awareness:\n", ROWS, COLS);
    mx = (float**)emalloc(sizeof(float*) * ROWS);
    pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * NUM_THREADS);
    int err = 0;
    printf("allocate %ld by %ld matrix\n", ROWS, COLS);
    for (int i = 0; i < NUM_THREADS; i++) {
      pthreads[i].thread_id = i;
      pthreads[i].job_size = ROWS / NUM_THREADS;
      pthreads[i].from = i * (ROWS / NUM_THREADS);
      if (pthreads[i].thread_id == NUM_THREADS -1) {
	pthreads[i].job_size += (ROWS % NUM_THREADS);
      }
      //      printf("thread_%d, is allocating mx[i] from %ld to %ld\n", pthreads[i].thread_id, pthreads[i].from, pthreads[i].from + pthreads[i].job_size);
      err = pthread_create(&pthreads[i].thread, NULL, normal_alloc_fn, (void *)&pthreads[i]);
      if (err != 0) {
	printf("error during thread creation, exit..\n");
	exit(-1);
      }
    }

    construct_barrier(pthreads, NUM_THREADS);
    
    pthread_barrier_init(&b, NULL, NUM_THREADS+1);
    /**measure access time*/

    for (int i = 0; i < NUM_THREADS; i++) {
      pthreads[i].thread_id = i;
      pthreads[i].job_size = ROWS / NUM_THREADS;
      pthreads[i].from = i * (ROWS / NUM_THREADS);
      if (pthreads[i].thread_id == NUM_THREADS - 1) {
	pthreads[i].job_size += (ROWS % NUM_THREADS);
      }
      err = pthread_create(&pthreads[i].thread, NULL, normal_access_fn, (void*)&pthreads[i]);
      if (err != 0) {
	printf("error during thread creation, exit...\n");
	exit(-1);
      }
    }
    
    pthread_barrier_wait(&b);
    auto start = tick();      
    construct_barrier(pthreads, NUM_THREADS);
    auto end = tick();
    printf("normal parallel access matrix, use: %lld\n", end - start);
    
    v = (float*)emalloc(sizeof(float) * COLS);
    w = (float*)emalloc(sizeof(float) * ROWS);
    for (int i = 0; i < COLS; i++) {
      v[i] = 1;
    }
    for (int i = 0; i < ROWS; i++) {
      w[i] = 0;
    }

    
    for (int i = 0; i < NUM_THREADS; i++) {
      pthreads[i].thread_id = i;
      pthreads[i].job_size = ROWS / NUM_THREADS;
      pthreads[i].from = i * (ROWS / NUM_THREADS);
      if (pthreads[i].thread_id == NUM_THREADS -1) {
	pthreads[i].job_size += (ROWS % NUM_THREADS);
      }
      err = pthread_create(&pthreads[i].thread, NULL, normal_multiply_fn, (void*)&pthreads[i]);
      if (err != 0) {
	printf("error during thread creation, exit...\n");
	exit(-1);
      }
    }
    pthread_barrier_wait(&b);
    start = tick();
    construct_barrier(pthreads, NUM_THREADS);
    end = tick();
    printf("normal parallel multiplication use: %lld\n", end - start);

    for (int i = 0; i < ROWS; i++) {
      free(mx[i]);
    }
    free(pthreads);
    free(mx);
    free(v);
    free(w);
    pthread_barrier_destroy(&b);
    auto done = std::chrono::high_resolution_clock::now();
    cout << "total execution time: " << std::chrono::duration_cast<std::chrono::duration<double> >(done - started).count() << " seconds" << endl;
    exit(0);


  } else if (with_numa == 2) {

    time_records = (long *)emalloc(sizeof(long) * NUM_THREADS);
    auto started = std::chrono::high_resolution_clock::now();
    printf("===Allocate mx with %ld * %ld, WITH NUMA awareness:\n", ROWS, COLS);
    mx = (float**)emalloc(sizeof(float*) * ROWS);
    pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * NUM_THREADS);
    int err = 0;
    printf("allocate %ld by %ld matrix\n", ROWS, COLS);

    for (int i = 0; i < NUM_THREADS; i++) {
      pthreads[i].thread_id = i;
      pthreads[i].job_size = ROWS / NUM_THREADS;
      pthreads[i].from = i * (ROWS / NUM_THREADS);
      if (pthreads[i].thread_id == NUM_THREADS -1) {
        pthreads[i].job_size += (ROWS % NUM_THREADS);
      }
      err = pthread_create(&pthreads[i].thread, NULL, alloc_fn, (void *)&pthreads[i]);
      if (err != 0) {
        printf("error during thread creation, exit..\n");
        exit(-1);
      }
    }
    construct_barrier(pthreads, NUM_THREADS);
    
    // evaluate access speed
    free(pthreads);
    pthreads = (pInfo*) emalloc(sizeof(*pthreads) * (NUM_THREADS - 1));
    pthread_barrier_init(&b, NULL, NUM_THREADS);
    for (int i = 1; i < NUM_THREADS; i++) {
      pthreads[i-1].thread_id = i;
      pthreads[i-1].job_size = ROWS / NUM_THREADS;
      pthreads[i-1].from = i * (ROWS / NUM_THREADS);
      if (pthreads[i-1].thread_id == NUM_THREADS - 1) {
        pthreads[i-1].job_size += (ROWS % NUM_THREADS);
      }

      err = pthread_create(&pthreads[i-1].thread, NULL, access_fn, (void*)&pthreads[i-1]);

      if (err != 0) {
        printf("error during thread creation, exit...\n");
        exit(-1);
      }
    }

    fix_current_thread_to_cpu(0);

    pthread_barrier_wait(&b);
    auto start = tick();
    do_access(0, 0, ROWS / NUM_THREADS);
    
    construct_barrier(pthreads, NUM_THREADS -1);
    auto end = tick();
    check_time_records(time_records, NUM_THREADS);
    printf("parallel access matrix with shape ROWS * COLS, use: %lld\n", end - start);
    check_normal_mx();
    
    v = (float*)emalloc(sizeof(float) * COLS);
    w = (float*)emalloc(sizeof(float) * ROWS);
    
    for (int i = 0; i < COLS; i++) {
      v[i] = 1;
    }
    for (int i = 0; i < ROWS; i++) {
      w[i] = 0;
    }


    // evaluate multiplication speed
    for (int i = 1; i < NUM_THREADS; i++) {
      pthreads[i-1].thread_id = i;
      // for the case in which rows < NUM_THREADS
      if (ROWS / NUM_THREADS == 0) {
        pthreads[i-1].from = i * 1;
        pthreads[i-1].job_size = 1;
      } else {
        pthreads[i-1].from  = i * (ROWS / NUM_THREADS);
        pthreads[i-1].job_size = ROWS / NUM_THREADS;
      }

      if (i == NUM_THREADS - 1) {
        pthreads[i-1].job_size += (ROWS % NUM_THREADS);
      }
      
      err = pthread_create(&pthreads[i-1].thread, NULL, multiply_fn, (void*)&pthreads[i-1]);

      if (err != 0) {
        printf("error during thread creation, exit...\n");
        exit(-1);
      }
    }
    

    fix_current_thread_to_cpu(0);
    pthread_barrier_wait(&b);
    do_multiplication(0, 0, ROWS / NUM_THREADS);
    
    construct_barrier(pthreads, NUM_THREADS-1);
    check_time_records(time_records, NUM_THREADS);

    check_normal_mx();
    check_w();


    int cpu_num = sched_getcpu();
    if (cpu_num) {
      printf("main thread is not fixed on cpu0 \n");
    }

    for (long i = 0; i < ROWS; i++) {
      numa_free(mx[i], sizeof(float) * COLS);
    }
    free(mx);
    free(pthreads);
    free(v);
    free(w);
    free(time_records);
    pthread_barrier_destroy(&b);
    auto done = std::chrono::high_resolution_clock::now();
    cout << "total execution time: " << std::chrono::duration_cast<std::chrono::duration<double> >(done - started).count() << " seconds" << endl;
    exit(0);


  } else {
    auto started = std::chrono::high_resolution_clock::now();
    mx = (float**)emalloc(sizeof(float*) * NUM_NODES);

    pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * NUM_NODES);
    pthread_t* thread; // array of pthread_t, for catching each created thread
    int err = 0;
    printf("===Allocate mx with %d * %ld, ", NUM_NODES, s / NUM_NODES);
    if (with_numa) {
      printf("WITH NUMA awareness\n");
    } else {
      printf("WITHOUT NUMA awareness\n");
    }
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

    pthreads =(pInfo*)remalloc(pthreads, sizeof(*pthreads) * NUM_THREADS);
    
    pthread_barrier_init(&b, NULL, NUM_THREADS+1);
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

    pthread_barrier_wait(&b);
    auto start = tick();
    construct_barrier(pthreads, NUM_THREADS);
    auto end = tick();
    //  auto done = std::chrono::high_resolution_clock::now();
    printf("parallel access matrix, use: %lld\n", end - start);
    //  std::cout << "By c++, parallel access matrix, use: " << std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count() << endl;
    //  check_mx();


    /**do the multiplication with a vector*/
    v = (float*)emalloc(sizeof(float) * COLS);
    w = (float*)emalloc(sizeof(float) * ROWS);
    for (int i = 0; i < COLS; i++) {
      v[i] = 1;
    }
    for (int i = 0; i < ROWS; i++) {
      w[i] = 0;
    }
  
  

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

    pthread_barrier_wait(&b);
    start = tick();
    construct_barrier(pthreads, NUM_THREADS);
    end = tick();
    //  done = std::chrono::high_resolution_clock::now();
    printf("parallel multiplication, use: %lld\n", end - start);
    //  std::cout << "By c++, parallel multiplication, use: " << std::chrono::duration_cast<std::chrono::nanoseconds>(done-started).count() << endl;
    //  check_w();
  
    int cpu_num = sched_getcpu();
    if (cpu_num) {
      printf("main thread is not fixed on cpu0 \n");
    }
    for (int i = 0; i < NUM_NODES; i++) {
      if (i == NUM_NODES - 1) {
	numa_free(mx[i], (s / NUM_NODES + s % NUM_NODES) * sizeof(float));
      } else {
	numa_free(mx[i], (s / NUM_NODES) * sizeof(float));
      }
    }
    pthread_barrier_destroy(&b);
    free(mx);
    free(pthreads);
    free(v);
    free(w);
    auto done = std::chrono::high_resolution_clock::now();
    cout << "total execution time: " << std::chrono::duration_cast<std::chrono::duration<double> >(done - started).count() << " seconds" << endl;
    exit(0);
  } // end of else
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

  //  printf("id: %d, thread: %lu is allocating %ld on node %d\n", info.thread_id, info.thread, info.job_size, info.row_in_mx);

  mx[info.thread_id] = (float*)numa_alloc_onnode(info.job_size * sizeof(float), info.row_in_mx);
  for (long k = 0; k < info.job_size; k++) {
    *(*(mx + info.row_in_mx) + k) = 0.0;
    //    mx[info.row_in_mx][k] = 0;
  }
  //  printf("thread %d  numa_alloc_onnode(%ld, %d), done\n", info.thread_id, info.job_size, info.row_in_mx);
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
    migrate_to_node(node_index);
    //    sched_cpu = sched_getcpu();
    //    current_on_node = topology[sched_cpu];
    //    printf("thread_id = %d, on cpu %d, on node %d\n", (unsigned long)info.thread_id, sched_cpu, current_on_node);
  }

  pthread_barrier_wait(&b);
  //  printf("id: %d, thread: %lu, is accessing on cpu: %d, on node: %d\n", info.thread_id, info.thread, sched_cpu, current_on_node);
  for (long i = info.from; i < info.from + info.job_size; i++) {
    mx[info.row_in_mx][i] = 3.0;
  }
  return (void *)0;
}

void* thr_multiply_fn(void* arg) {
  pid_t pid = getpid();
  int sched_cpu = sched_getcpu();
  int current_on_node = topology[sched_cpu];

  pInfo info = *(pInfo*)arg;
  int node_index = info.thread_id / (NUM_THREADS / NUM_NODES);
  if(with_numa && node_index != current_on_node) {
    //printf("thread_id = %d, on cpu %d, on node %d, should be on node%d\n", (unsigned long)info.thread_id, sched_cpu, current_on_node, node_index);
    migrate_to_node(node_index);
  }
  long each_size = s / NUM_NODES;
  pthread_barrier_wait(&b);
  //  printf("id: %d, thread: %lu, is doing multiplication from cpu %d, on node %d\n", info.thread_id, info.thread, sched_cpu, current_on_node);
  for (long r = info.from; r < info.from  + info.job_size; ++r) {
    // get corresponding mx(i, j)
    for (long i = 0; i < COLS; i++) {
      long l = r * COLS + i;
      long row_in_mx = l / each_size - 1;
      if (row_in_mx < 0) {
	row_in_mx = 0;
      }
      long col_in_mx = l % each_size;
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
  int thread_id = 0;
  vector<int> index_set = {0, 32, 2, 34, 3, 35, 1, 33};
  for (int node = 0 ; node < index_set.size(); node++) {
    vector<int> cpus_on_node;
    for (int cpu = index_set[node]; cpu < index_set[node] + 32; cpu+=4) {
      //      printf("make pair: %d <=> %d\n", node, cpu);
      topology.insert(make_pair(cpu, node));
      cpus_on_node.push_back(cpu);
      distribute_scheme.insert(make_pair(thread_id, cpu));
      thread_id += 1;
    }
    topology_inverse.insert(make_pair(node, cpus_on_node));
  }
}


void* normal_alloc_fn(void* arg) {
  pInfo info = *(pInfo*)arg;
  for (long k = info.from; k < info.from + info.job_size; k++) {
    mx[k] = (float*)emalloc(sizeof(float) * COLS);
    for (long j = 0; j < COLS; j++){
      mx[k][j] = 0;
    }
  }
  //  printf("thread_%d, done alloc.\n", info.thread_id);
  return (void*) 0;
}

void* normal_access_fn(void* arg) {
  pInfo info = *(pInfo*)arg;

  pthread_barrier_wait(&b);
  for (long k = info.from; k < info.from + info.job_size; k++) {
    for (long j = 0; j < COLS; j++) {
      mx[k][j] = 3;
    }
  }
}


void* normal_multiply_fn(void* arg) {
  pInfo info = *(pInfo*)arg;
  //  printf("thread_id = %d, from = %d, to = %d\n", info.thread_id, info.from, info.from + info.job_size);
  pthread_barrier_wait(&b);
  for (long k = info.from; k < info.from + info.job_size; k++) {
    for (long j = 0; j < COLS; j++) {
      w[k] += (mx[k][j] * v[j]);
    }
  }
}

void* alloc_fn(void* arg) {
  pInfo info = *(pInfo *)arg;
  int node_index = info.thread_id / (NUM_THREADS / NUM_NODES);
  
  for (long k = info.from; k < info.from + info.job_size; k++) {
    mx[k] = (float*)numa_alloc_onnode(sizeof(float) * COLS, node_index);
    for (long j = 0; j < COLS; j++){
      mx[k][j] = 0;
    }
  }
}

void do_access(int thread_id, long from, long job_size) {
  int current_cpu = sched_getcpu();
  auto start = tick();

  for (long k = from; k < (from + job_size); k++) {
    for (long j = 0; j < COLS; j++) {
      mx[k][j] = 3;
    }
  }

  auto end = tick();
  time_records[thread_id] = end - start;

  if (current_cpu != sched_getcpu()) {
    printf("during access, cpus is not fixed!\n");
  }
}


void check_time_records(long* time_records, int n) {
  int index = 0;
  long max = 0;

  for (int i = 0; i < n; i++) {
    printf("thread_%d takes: %ld\n", i, time_records[i]);
    if (time_records[i] > max) {
      index = i;
      max = time_records[i];
    }
  }
  printf("thread_%d takes longest, use: %ld\n", index, max);
}

void* access_fn(void* arg) {
  pInfo info = *(pInfo*)arg;
  
  fix_current_thread_to_cpu(distribute_scheme[info.thread_id]);

  pthread_barrier_wait(&b);
  do_access(info.thread_id, info.from, info.job_size);

  return (void *)0;
}

void* multiply_fn(void* arg) {
  pInfo info = *(pInfo*)arg;

  fix_current_thread_to_cpu(distribute_scheme[info.thread_id]);

  pthread_barrier_wait(&b);
  do_multiplication(info.thread_id, info.from, info.job_size);
  //  printf("job_size = %ld, thread_%d\n", info.job_size, info.thread_id);

  return (void *)0;
}

void migrate_to_node(int node_index) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  vector<int> cpus = topology_inverse[node_index];
  for (int i = 0; i < cpus.size(); i++) {
    CPU_SET(cpus[i], &cpuset);
  }
  pthread_t tid = pthread_self();
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
}

void do_multiplication(int thread_id, long from, long job_size) {
  int current_cpu = sched_getcpu();
  auto start = tick();

  for (long k = from; k < (from + job_size); k++) {
    for (long j = 0; j < COLS; j++) {
      printf("mx[%ld][%ld] = %ld, v[%ld] = %ld\n", k, j, mx[k][j], j, v[j]);
            w[k] += (mx[k][j] * v[j]);
      //      mx[k][j] = 4;
    }
  }

  auto end = tick();
  time_records[thread_id] = end - start;
  if (current_cpu != sched_getcpu() || current_cpu != distribute_scheme[thread_id]) {
    printf("during multiplication, cpu is not fixed!\n");
  }
}

void fix_current_thread_to_cpu(int num) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(num, &cpuset);
  pthread_t tid = pthread_self();
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
}


void check_normal_mx() {
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      printf("%d, ", mx[i][j]);
    }
  }
}
