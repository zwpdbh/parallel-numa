#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <sched.h>
#include <stdio.h>
#include <numa.h>
#include <stdlib.h>
#include <string>
#include <map>

using namespace std;

void *emalloc(size_t s) {
  void *result = malloc(s);
  if (result == NULL) {
    fprintf(stderr, "memory allocation failed");
    exit(EXIT_FAILURE);
  }
  return result;
}

void initMap(map<int, int> &topology) {
  // node 0 cpus: 0 4 8 12 16 20 24 28
  for (int i = 0; i <=28; i+=4) {
    topology.insert(make_pair(i, 0));
  }

  // node 1 cpus: 32 36 40 44 48 52 56 60
  for (int i = 32; i <=60; i+=4) {
    topology.insert(make_pair(i, 1));
  }

  // node 2 cpus: 2 6 10 14 18 22 26 30
  for (int i = 2; i <= 30; i+=4) {
    topology.insert(make_pair(i, 2));
  }

  // node 3 cpus: 34 38 42 46 50 54 58 62
  for (int i = 34; i <= 62; i+=4) {
    topology.insert(make_pair(i, 3));
  }
  
  // node 4 cpus: 3 7 11 15 19 23 27 31
  for (int i = 3; i <= 31; i+=4) {
    topology.insert(make_pair(i, 4));
  }

  // node 5 cpus: 35 39 43 47 51 55 59 63
  for (int i = 35; i <= 63; i+=4) {
    topology.insert(make_pair(i, 5));
  }

  // node 6 cpus: 1 5 9 13 17 21 25 29
  for (int i = 1; i <= 29; i+=4) {
    topology.insert(make_pair(i, 6));
  }

  // node 7 cpus: 33 37 41 45 49 53 57 61
  for (int i = 33; i <= 61; i+=4) {
    topology.insert(make_pair(i, 7));
  }
}

int main(int argc, char* argv[]) {
  if (numa_available() < 0) {
    cout << "Your system does not support NUMA API" << endl;
    exit(-1);
  } else {
    cout << "The number of highest possible node in the system is: " << numa_max_possible_node()  << endl;
    
  }


  if (argc < 3) {
    cout << "Please specify m n on command line" << endl;
    exit(-1);
  }
  long m, n;
  int n_cpu;
  m = stol(argv[1]);
  n = stol(argv[2]);
  n_cpu = stoi(argv[3]);


  //  float* a = (float*)emalloc(n * sizeof(float));
  vector<float*> a;
  vector<float*> b;
  vector<float*> c;

  long i, j;



  float* a_each = NULL;  
  float* eachRow = NULL;
  float* c_each = NULL;

  char* binding_topology = getenv("OMP_PLACES");
  if (binding_topology != NULL) {
    printf("OMP_PLACES=%s\n", binding_topology);
  }

  map<int, int> topology;
  initMap(topology);





#pragma omp parallel proc_bind(close) num_threads(n_cpu) default(none) shared(m, n, a, b, c, topology) private(i, j, eachRow, a_each, c_each)
  {
    int thread_num = omp_get_thread_num();
    int cpu_num = sched_getcpu();

    printf("Thread %3d is running on CPU %3d, on node %d\n", thread_num, cpu_num, topology[cpu_num]);
    
#pragma omp for ordered schedule (static)
    for (i = 0; i < m; i++) {

      eachRow = NULL;
      a_each = NULL;
      c_each = NULL;

      cpu_num = sched_getcpu();
      int which_node = topology[cpu_num];

      eachRow = (float*)numa_alloc_onnode(n * sizeof(float), which_node);
      a_each = (float*)numa_alloc_onnode(sizeof(float), which_node);
      c_each = (float*)numa_alloc_onnode(sizeof(float), which_node);

      if (eachRow == NULL || a_each == NULL || c_each == NULL) {
	printf("error during allocation numa memory on node %d\n", cpu_num);
	exit(-1);
      }

      a_each[0] = 0.0;
      c_each[0] = 2.0;
      for (j = 0; j < n; j++) {
	eachRow[j] = 2.0;
      }

#pragma omp ordered
      a.push_back(a_each);
#pragma omp ordered
      b.push_back(eachRow);
#pragma omp ordered
      c.push_back(c_each);
    }    
  }

  printf("Check if the thread is paralleled as planed\n");
  auto started = std::chrono::high_resolution_clock::now();
#pragma omp parallel proc_bind(close) num_threads(n_cpu) default(none) shared(a,b,c,m,n, topology) private(i, j)
  {
    int thread_num = omp_get_thread_num();
    int cpu_num = sched_getcpu();

    printf("Thread %3d is running on CPU %3d, on node %d\n", thread_num, cpu_num, topology[cpu_num]);

#pragma omp for schedule (static)
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
	a[i][0] += (b[i][j] * (*c[i]));
      }
    }  
  }

  auto done = std::chrono::high_resolution_clock::now();
  std::cout << "From initialization to finished, use: " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << "ms" << endl;


  for (i = 0; i < m; i++) {
    numa_free(b[i], n * sizeof(float));
    numa_free(a[i], sizeof(float));
    numa_free(a[i], sizeof(float));
  }


  return 0;
}
