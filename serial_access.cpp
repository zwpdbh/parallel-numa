#include <iostream>
#include <vector>
#include <sched.h>
#include <stdio.h>
#include <numa.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>


using namespace std;

void *emalloc(size_t s);
void *remalloc(void *p, size_t s);
void initMap(map<int, int> &topology, map<int, vector<int> > &topology_inverse);

inline uint64_t tick() {
  uint32_t tmp[2];
  __asm__ ("rdtsc" : "=a" (tmp[1]), "=d" (tmp[0]) : "c" (0x10) );
  return (((uint64_t) tmp[0]) << 32) | tmp[1];
}

/**allocate and access on the same node*/
int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("usage: <size> <remote=1 true, or 0 false>, exit...\n");
    exit(-1);
  }

  if(numa_available() == -1) {
    printf("no libnuma support\n");
  } else {
    numa_set_strict(1);
  }
  
  map<int, int> topology;
  map<int, vector<int> > topology_inverse;
  initMap(topology, topology_inverse);

  long mx_size = atol(argv[1]);
  int remote = atoi(argv[2]);
  int on_cpu = 0;
  int on_node = 0;
  pid_t pid;
  pthread_t tid;

  pid = getpid();
  tid = pthread_self();
  printf("main thread: pid %lu tid %lu, on cpu: %d\n", (unsigned long)pid, (unsigned long)tid, sched_getcpu());
  
  cpu_set_t cpuset;
  cpu_set_t allcpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(on_cpu, &cpuset);
  pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
  int cpu_num = sched_getcpu();
  printf("Now, main thread tid = %lu, is executing on cpu: %d, on node %d\n",pthread_self(), cpu_num, topology[cpu_num]);

  float* mx = NULL;
  
  if (remote) {
    on_node = 7;
  }
  printf("will allocate memory on node %d\n", on_node);
  mx = (float*)numa_alloc_onnode(mx_size * mx_size * sizeof(float), on_node);
  if (mx == NULL) {
    printf("could not allocate memory on node %d, exit...\n", on_node);
  }
  
  auto start = tick();
  for (long i = 0; i < mx_size * mx_size; i++) {
    mx[i] = i + i * 0.5 + i / 2;
  }

  auto end = tick();
  if (remote && topology[sched_getcpu()] != on_node) {
    std::cout << "remote access, use: " << end - start << endl;
  } else if (!remote && topology[sched_getcpu()] == on_node) {
    std::cout << "local access, use: " << end - start << endl;
  }

  numa_free(mx, mx_size * mx_size * sizeof(float));

  return 0;
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
