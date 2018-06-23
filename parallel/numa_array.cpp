#include "mylib.h"

int correct = 0;

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
  printf("correct allocated elements are %d\n", correct);
}

int main() {
  int s = 12800;
  int i = 0;
  float* v;
  int size_for_each_node = s / 8;
  int y = 0;
  int distribution[8];
  numa_set_strict(1); 

  for (i = 0; i < 8; i++) {
    distribution[i] = 0;
  }

  //v = (float*)numa_alloc_interleaved(sizeof(float) * s);
  v = (float*)numa_alloc_onnode(sizeof(float) * s, 0);
  
  int status[1];
  void* pa;
  pa = (void*)(v);
  int to_node[1];
  to_node[0] = 6;
  pid_t p = getpid();

  for (i = 0; i < s; i++) {
    v[i] = 0;
  }

  numa_move_pages(p, 100, &pa, to_node, status, MPOL_MF_MOVE);

  for (i = 0; i < s; i++) {
    distribution_on_node(i / size_for_each_node, &v[i], distribution);
  }

  printf("\n");
  check_distribution(distribution, s);

  numa_free(v, sizeof(float) * s);
  return 0;
}
