/*
 * rbc_index.h
 *
 *  Created on: Oct 9, 2013
 *      Author: ryan
 */

#ifndef RBC_INDEX_H_
#define RBC_INDEX_H_

#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include "rbc/utils.h"
#include "rbc/rbc.h"
#include "rbc/dists.h"
#include <gsl/gsl_sort.h>
using namespace flann;

#include "common.h"
#include "sdc_index.h"

class RBCIndex {
public:
	RBCIndex(float*, int, int);
	virtual ~RBCIndex();
	void buildIndex(int);
	void knnSearch(Matrix<float>&, Matrix<int>&, Matrix<float>&, int,
			SearchParams&);

private:
	void mysearchOneShotK(matrix, matrix, matrix, rep*, unint**, real**, unint);
	void mybrutePar(matrix, matrix, unint*, real*, real*);
	void mybruteMapK(matrix, matrix, rep*, unint*, unint**, real**, unint,
			real*);
	float* ddata;
	int drows, dcols;
	matrix x, r;
	rep *ri;
	unint* shuf;
#ifdef EST
	SDCIndex *est_index;
	int cluster_size;
#endif
};

#endif /* RBC_INDEX_H_ */
