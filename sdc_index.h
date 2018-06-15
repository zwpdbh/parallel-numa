/*
 * ckdtree.h
 *
 *  Created on: Jul 22, 2013
 *      Author: xiaoxintang
 */

#ifndef CKDTREE_H_
#define CKDTREE_H_

#include "sdc_kmeans.h"
#include "common.h"
#include <vector>

using namespace std;

#define ITER	10
#define TRAINT	10

#define SCALE_U 0.2
#define SCALE_D 0.6

class SDCIndex {
public:
	SDCIndex(float*, int, int);
	virtual ~SDCIndex();
	void buildIndex();
	void buildIndex(int, int, int);
	void knnSearch(Matrix<float>&, Matrix<int>&, Matrix<float>&, int,
			SearchParams&);
	void update(float pre, float precision);
	void getStatics(int, int, int, float*, float*, int, int);

	float* init(float*, float* md = NULL);
	int getClusterSize();
	float estimate(int, float*);
	float estimate(unsigned char*, float*);
	void explore(float*, ResultSet<float>*, float*);

	int drows, dcols, branch;
	vector<Cluster*> clusters;

	int num, total;
	static float scale_u, scale_d;
	static int DEPTH_, BRANCH_;

private:
	float* ddata;
	unsigned char* beLongTo;

	static L2<float> distance_;
};

#endif /* CKDTREE_H_ */
