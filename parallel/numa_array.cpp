#include "mylib.h"
#include <omp.h>

using namespace std;

std::map<int, int> cpu_to_node; 
std::map<int, int> thread_to_cpu; 
std::map<int, std::vector<int> > node_to_cpus;

int correct = 0;

void move_one_page_on_node(int node_id, float* v) {
  pid_t p = getpid();
  int to_node[1];
  to_node[0] = node_id;
  int status[1];
  //  void* page_address = (void*)((uintptr_t)v & ~0xFFF);
  void* page_address =(void*)v;
  numa_move_pages(p, 1, &page_address, to_node, status, MPOL_MF_MOVE);
}

void distribute_pages(float* v, int s) {
  int size_on_each_node = s / 8;
  int node_id = 0;
  int interval = 4096 / (sizeof(float));
  int num_of_page = 1;

  move_one_page_on_node(node_id, v);
  int count = 0;

  for (int i = 0; i < s; i++) {
    if (count == (sizeof(float) - 2)) {
      node_id = i / size_on_each_node;
      move_one_page_on_node(node_id, v+i);
      count = 0;
    }
    count += 1;
  }
} 


void distribution_on_node(int on_node, void* address, int* distribution) {
  void* ptr_to_check = address;
  int status[1];
  int ret_code;
  status[0] = -1;
  ret_code = move_pages(0, 1, &ptr_to_check, NULL, status, 0);
  
  distribution[status[0]] = distribution[status[0]] + 1;;
  if(on_node == status[0]) {
    correct += 1;
  }
}

void check_distribution(int* distribution, int s) {
  for(int i = 0; i < 8; i++) {
    printf("node %d = %d, %.2f\n", i, distribution[i], ((float)distribution[i] / (float)s));
  }
  printf("correct allocated elements are %d, = %.2f\n", correct, (float)correct/12800);
}

void first_touch_serial(float *v, int s, int migrate) {
  int current_node = 0;
  int which_node;
  int size_on_each_node = s / 8;
  migrate_to_node(current_node, node_to_cpus);
  for (int i = 0; i < s; i++) {
    if (migrate) {
      which_node = i / size_on_each_node;
      if (current_node != which_node) {
	printf("from i = %d, first touch on node %d\n", i, which_node);
	current_node = which_node;
	migrate_to_node(which_node, node_to_cpus);
      }
    }
    v[i] = 0;
  }
}

void first_touch_parallel(float *v, int s) {
  int i;

#pragma omp parllel proc_bind(close) num_threads(g->n_threads) \
  default(none) shared(v,s) private(i)
  {
    int sched_cpu = sched_getcpu();
    int thread_id = omp_get_thread_num();
    if (thread_to_cpu[thread_id] != sched_cpu) {
      printf("thread_%d should execute on cpu_%d, but it is running on %d\n", \
	     thread_id, thread_to_cpu[thread_id], sched_cpu);
    }
#pragma omp for schedule (static)
    for (i = 0; i < s; i++) {
      v[i] = 0;
    }
  }
}

int main(int argc, char* argv[]) {
  int thread_id = 0;
  int sched_cpu;
  vector<int> index_set = {0, 32, 2, 34, 3, 35, 1, 33};
  for (int node = 0; node < index_set.size(); node++) {
    vector<int> cpus_on_node;
    for (int cpu = index_set[node]; cpu < index_set[node] + 32; cpu+= 4) {
      cpu_to_node.insert(make_pair(cpu, node));
      
      thread_to_cpu.insert(make_pair(thread_id, cpu));
      thread_id += 1;
      cpus_on_node.push_back(cpu);
    }
    node_to_cpus.insert(make_pair(node, cpus_on_node));
  }

  int s = 12800;
  float* v;
  int y = 0;
  int distribution[8];

  numa_set_strict(1); 

  for (int i = 0; i < 8; i++) {
    distribution[i] = 0;
  }

  int alloc_option = 0;
  if (argc == 2) {
    alloc_option = atoi(argv[1]);
    printf("alloc_option is: %d\n", alloc_option);
  }
  

  if (alloc_option == 0) {
    printf("using emalloc\n");
    v = (float*)emalloc(sizeof(float) * s);
  } else if (alloc_option == 1) {
    printf("using numa_alloc_onnode, on node 0\n");
    v = (float*)numa_alloc_onnode(sizeof(float) * s, 0);
  } else if (alloc_option == 2) {
    printf("using numa_alloc_interleaved\n");
    v = (float*)numa_alloc_interleaved(sizeof(float) * s);
  }


  first_touch_serial(v, s, 0);
  //first_touch_parallel(v,s);
  
  //  move_one_page_on_node(7, v, 1);
  //  move_one_page_on_node(6, v + 2000, 2);

  distribute_pages(v, s);

  for (int i = 0; i < s; i++) {
    distribution_on_node(i / (s / 8), &v[i], distribution);
  }

  printf("\n");
  check_distribution(distribution, s);

  if (alloc_option == 0) {
    free(v);
  } else {
    numa_free(v, sizeof(float) * s);
  }

  return 0;
}
