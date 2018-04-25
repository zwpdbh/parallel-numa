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

  if (argc < 3) {
    cout << "Please specify m n on command line" << endl;
    exit(-1);
  }
  long m, n;
  int n_cpu;
  m = stol(argv[1]);
  n = stol(argv[2]);
  n_cpu = stoi(argv[3]);


  float* a = (float*)emalloc(n * sizeof(float));
  vector<float*> b;
  float* c = (float*)emalloc(n * sizeof(float));

  long i, j;


  
  float* eachRow = NULL;

  char* binding_topology = getenv("OMP_PLACES");
  if (binding_topology != NULL) {
    printf("OMP_PLACES=%s\n", binding_topology);
  }

  map<int, int> topology;
  initMap(topology);

#pragma omp parallel proc_bind(close) num_threads(n_cpu) default(none) shared(m, n, b, c, topology) private(i, j, eachRow)
  {
    int thread_num = omp_get_thread_num();
    int cpu_num = sched_getcpu();
    printf("Thread %3d is running on CPU %3d, on node %d\n", thread_num, cpu_num, topology[cpu_num]);
    
#pragma omp for ordered schedule (static)
    for (i = 0; i < m; i++) {
      eachRow = (float*)emalloc(n * sizeof(float));

      if (eachRow == NULL) {
	printf("error during allocation numa memory on node %d\n", cpu_num);
	exit(-1);
      }

      for (j = 0; j < n; j++) {
	eachRow[j] = 2.0;
      }
#pragma omp ordered
      b.push_back(eachRow);
    }

#pragma omp for schedule (static)
    for (i = 0; i < n; i++) {
      c[i] = 2.0;
    }
  }

  auto started = std::chrono::high_resolution_clock::now();
      
#pragma omp parallel proc_bind(close) num_threads(n_cpu) default(none) shared(a,b,c,m,n) private(i, j) 
  {

#pragma omp for schedule (static)
    for (i = 0; i < m; i++) {
      a[i] = 0.0;
      for (j = 0; j < n; j++) {
	a[i] += (b[i][j] * c[j]);
      }
    }  
  }

  auto done = std::chrono::high_resolution_clock::now();
  std::cout << "From initialization to finished, use: " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << "ms" << endl;

  for (i = 0; i < m; i++) {
    free(b[i]);
  }


  free(c);
  free(a);




  return 0;
}
