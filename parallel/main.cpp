#include "pthread_parallel.h"
#include "openmp_parallel.h"

int main(int argc, char* argv[]) {
  if (argc !=4 ) {
    printf("usage: ./bin/parallel-numa <rows> <cols> <options>, exit...\n");
    printf("options:\n");
    printf("1, pthread with numa\n");
  }


  if (numa_available() == -1) {
    printf("no libnuma available\n");
  } else {
    numa_set_strict(1);
  }

  long rows = atol(argv[1]);
  long cols = atol(argv[2]);
  int option = atoi(argv[3]);
  int n_threads = 64;
  int n_nodes = 8;
  int pid;

  fix_current_thread_to_cpu(0);

  if (option == 0) {
    printf("pthread without numa:\n");
    GlobalVar* g = new GlobalVar(rows, cols, n_threads, n_nodes, option);
    pthread_without_numa(g);
    delete g;
  } else if (option == 1) {
    printf("pthread with numa:\n");
    GlobalVar* g = new GlobalVar(rows, cols, n_threads, n_nodes, option);
    pthread_with_numa(g);
    delete g;
  } else if (option == 4) {
    printf("pthread with numa and aligned:\n");
    GlobalVar* g = new GlobalVar(rows, cols, n_threads, n_nodes, option);
    pthread_with_numa_aligned(g);
    delete g;
  } else if (option == 2) {
    printf("openmp with numa:\n");
    if ((pid = vfork()) < 0) {
      printf("error during fork to change environment\n");
    } else if (pid == 0) {
      // use exec to do change the environment
      _exit(0);
    } 

    GlobalVar* g = new GlobalVar(rows, cols, n_threads, n_nodes, option);
    openmp_with_numa(g);
    delete g;
  } else if (option == 3) {
    printf("openmp without numa:\n");
    GlobalVar* g = new GlobalVar(rows, cols, n_threads, n_nodes, option);
    openmp_without_numa(g);
    delete g;
  } else {
    printf("unvalid option, exit...\n");
  } 

  return 0;
}
