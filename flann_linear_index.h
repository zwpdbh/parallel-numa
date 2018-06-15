/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef FLANN_LINEAR_INDEX_H_
#define FLANN_LINEAR_INDEX_H_

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"

#include "common.h"
#include "sdc_index.h"

namespace flann {

struct LinearIndex_mParams: public IndexParams {
	LinearIndex_mParams() {
		(*this)["algorithm"] = FLANN_INDEX_LINEAR;
	}
};

template<typename Distance>
class LinearIndex_m: public NNIndex<Distance> {
public:

	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;
	typedef NNIndex<Distance> BaseClass;

	LinearIndex_m(const IndexParams& params = LinearIndex_mParams(),
			Distance d = Distance()) :
			BaseClass(params, d) {
#ifdef EST
		est_index = NULL;
#endif
	}

	LinearIndex_m(const Matrix<ElementType>& input_data,
			const IndexParams& params = LinearIndex_mParams(), Distance d =
					Distance()) :
			BaseClass(params, d) {
		setDataset(input_data);
#ifdef EST
		est_index = NULL;
#endif
	}

	LinearIndex_m(const LinearIndex_m& other) :
			BaseClass(other) {
#ifdef EST
		est_index = NULL;
#endif
	}

	LinearIndex_m& operator=(LinearIndex_m other) {
		this->swap(other);
		return *this;
	}

	virtual ~LinearIndex_m() {
	}

	BaseClass* clone() const {
		return new LinearIndex_m(*this);
	}

	void addPoints(const Matrix<ElementType>& points, float rebuild_threshold =
			2) {
		assert(points.cols==veclen_);
		extendDataset(points);
	}

	flann_algorithm_t getType() const {
		return FLANN_INDEX_LINEAR;
	}

	int usedMemory() const {
		return 0;
	}

	template<typename Archive>
	void serialize(Archive& ar) {
		ar.setObject(this);

		ar & *static_cast<NNIndex<Distance>*>(this);

		if (Archive::is_loading::value) {
			index_params_["algorithm"] = getType();
		}
	}

	void saveIndex(FILE* stream) {
		serialization::SaveArchive sa(stream);
		sa & *this;
	}

	void loadIndex(FILE* stream) {
		serialization::LoadArchive la(stream);
		la & *this;
	}

	void findNeighbors(ResultSet<DistanceType>& resultSet,
			const ElementType* vec,
			const SearchParams& /*searchParams*/) const {
#ifdef EST
		float *mdist = NULL;
		mdist = est_index->init((float*) vec);
#endif
		if (removed_) {
			DistanceType worst = resultSet.worstDist();
			for (size_t i = 0; i < points_.size(); ++i) {
				if (removed_points_.test(i))
					continue;
				DistanceType dist = 0;
#ifdef EST
//				++est_index->total;
				dist = est_index->estimate(i, mdist);
				if (dist >= worst) {
//					++est_index->num;
					continue;
				}
#endif
				dist = distance_(points_[i], vec, veclen_
//						,resultSet.worstDist()
						);
				resultSet.addPoint(dist, i);
				worst = resultSet.worstDist();
			}
		} else {
			DistanceType worst = resultSet.worstDist();
			for (size_t i = 0; i < points_.size(); ++i) {
				DistanceType dist = 0;
#ifdef EST
//				++est_index->total;
				dist = est_index->estimate(i, mdist);
				if (dist >= worst) {
//					++est_index->num;
					continue;
				}
#endif
				dist = distance_(points_[i], vec, veclen_
//						,resultSet.worstDist()
						);
				resultSet.addPoint(dist, i);
				worst = resultSet.worstDist();
			}
		}

#ifdef EST
//		if (est_index->total > 1000000) {
//			printf("%.4f\n", est_index->num * 1.0f / est_index->total);
//			est_index->num = est_index->total = 0;
//		}
		delete[] mdist;
#endif
	}
protected:
#ifdef EST
	SDCIndex *est_index;
#endif

	void buildIndexImpl() {
		/* nothing to do here for linear search */
#ifdef EST
		est_index = new SDCIndex(points_[0], size_, veclen_);
		est_index->buildIndex();
#endif
	}

	void freeIndex() {
		/* nothing to do here for linear search */
	}

private:

	USING_BASECLASS_SYMBOLS};

	}

#endif // FLANN_LINEAR_INDEX_H_
