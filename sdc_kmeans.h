#ifndef FLANN_KMEANS_H_
#define FLANN_KMEANS_H_

#include <algorithm>
#include <string>
#include <map>
#include <cassert>
#include <limits>
#include <cmath>

#include "flann/general.h"
#include "flann/algorithms/nn_index.h"
#include "flann/algorithms/dist.h"
#include "flann/util/random.h"
#include "flann/algorithms/center_chooser.h"
#include "flann/util/matrix.h"
#include "flann/util/result_set.h"
#include "flann/util/heap.h"
#include "flann/util/allocator.h"
#include "flann/util/saving.h"
#include "flann/util/logger.h"

using namespace flann;

struct KMeansParams: public IndexParams {
	KMeansParams(int branching = 64, int iterations = 11,
			flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
			float cb_index = 0.2) {
		(*this)["algorithm"] = FLANN_INDEX_KMEANS;
		// branching factor
		(*this)["branching"] = branching;
		// max iterations to perform in one kmeans clustering (kmeans tree)
		(*this)["iterations"] = iterations;
		// algorithm used for picking the initial cluster centers for kmeans tree
		(*this)["centers_init"] = centers_init;
		// cluster boundary index. Used when searching the kmeans tree
		(*this)["cb_index"] = cb_index;
	}
};

struct Cluster {
	float *centers, *radiuses, *variances, *averages;
	int *num, *beLongTo;
	int rows, cols, start;
	float radius, variance, average;
	~Cluster() {
		delete[] centers;
		delete[] radiuses;
		delete[] variances;
		delete[] averages;
		delete[] beLongTo;
		delete[] num;
	}
};

/**
 * Hierarchical kmeans index
 *
 * Contains a tree constructed through a hierarchical kmeans clustering
 * and other information for indexing a set of points for nearest-neighbour matching.
 */
template<typename Distance>
class KMeans: public NNIndex<Distance>
{
public:
	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;
	typedef NNIndex<Distance> BaseClass;
	typedef bool needs_vector_space_distance;

	flann_algorithm_t getType() const {
		return FLANN_INDEX_KMEANS;
	}

	/**
	 * Index constructor
	 *
	 * Params:
	 *          inputData = dataset with the input features
	 *          params = parameters passed to the hierarchical k-means algorithm
	 */
	KMeans(const Matrix<ElementType>& inputData, const IndexParams& params =
			KMeansParams(), Distance d = Distance()) :
			BaseClass(params, d), root_(NULL), memoryCounter_(0) {
		branching_ = get_param(params, "branching", 32);
		iterations_ = get_param(params, "iterations", 11);
		if (iterations_ < 0) {
			iterations_ = (std::numeric_limits<int>::max)();
		}
		centers_init_ = get_param(params, "centers_init", FLANN_CENTERS_RANDOM);
		cb_index_ = get_param(params, "cb_index", 0.4f);

		initCenterChooser();
		chooseCenters_->setDataset(inputData);
		setDataset(inputData);
	}

	/**
	 * Index destructor.
	 *
	 * Release the memory used by the index.
	 */
	virtual ~KMeans() {
		delete chooseCenters_;
		freeIndex();
	}

	/**
	 * Computes the inde memory usage
	 * Returns: memory used by the index
	 */
	int usedMemory() const {
		return pool_.usedMemory + pool_.wastedMemory + memoryCounter_;
	}

	Cluster* getCluster() {
		Cluster* c = new Cluster();
		c->rows = root_->childs.size();
		c->cols = veclen_;
		c->centers = new float[c->rows * c->cols];
		c->beLongTo = new int[size_];
		c->radiuses = new float[c->rows];
		c->variances = new float[c->rows];
		c->averages = new float[c->rows];
		c->num = new int[c->rows];

		int cur = 0, size = veclen_ * sizeof(float);
		for (int i = 0; i < c->rows; ++i) {
			NodePtr child = root_->childs[i];
			memcpy(c->centers + cur, child->pivot, size);
			cur += veclen_;
			for (size_t j = 0; j < child->points.size(); ++j) {
				c->beLongTo[child->points[j].index] = i;
			}
			c->radiuses[i] = child->radius;
			c->variances[i] = child->variance;
			c->averages[i] = child->average;
			c->num[i] = child->points.size();
		}
		c->radius = root_->radius;
		c->variance = root_->variance;
		c->average = root_->average;
		return c;
	}

	using BaseClass::buildIndex;

	void saveIndex(FILE* stream) {

	}

	void loadIndex(FILE* stream) {

	}

	void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec,
			const SearchParams& searchParams) const {

	}

	BaseClass* clone() const {
		return NULL;
	}

protected:
	/**
	 * Builds the index
	 */
	void buildIndexImpl() {
		if (branching_ < 2) {
			throw FLANNException("Branching factor must be at least 2");
		}

		std::vector<int> indices(size_);
		for (size_t i = 0; i < size_; ++i) {
			indices[i] = int(i);
		}

		root_ = new (pool_) Node();
		computeNodeStatistics(root_, indices);
		computeClustering(root_, &indices[0], (int) size_, branching_);
	}

private:
	struct PointInfo {
		size_t index;
		ElementType* point;
	};

	/**
	 * Struture representing a node in the hierarchical k-means tree.
	 */
	struct Node {
		/**
		 * The cluster center.
		 */
		DistanceType* pivot;
		/**
		 * The cluster radius.
		 */
		DistanceType radius;
		/**
		 * The cluster variance.
		 */
		DistanceType variance;

		DistanceType average;
		/**
		 * The cluster size (number of points in the cluster)
		 */
		int size;
		/**
		 * Child nodes (only for non-terminal nodes)
		 */
		std::vector<Node*> childs;
		/**
		 * Node points (only for terminal nodes)
		 */
		std::vector<PointInfo> points;
		/**
		 * Level
		 */
//        int level;
		~Node() {
			delete[] pivot;
			if (!childs.empty()) {
				for (size_t i = 0; i < childs.size(); ++i) {
					childs[i]->~Node();
				}
			}
		}
	};
	typedef Node* NodePtr;

	/**
	 * Alias definition for a nicer syntax.
	 */
	typedef BranchStruct<NodePtr, DistanceType> BranchSt;

	void initCenterChooser() {
		switch (centers_init_) {
		case FLANN_CENTERS_RANDOM:
			chooseCenters_ = new RandomCenterChooser<Distance>(distance_);
			break;
		case FLANN_CENTERS_GONZALES:
			chooseCenters_ = new GonzalesCenterChooser<Distance>(distance_);
			break;
		case FLANN_CENTERS_KMEANSPP:
			chooseCenters_ = new KMeansppCenterChooser<Distance>(distance_);
			break;
		default:
			throw FLANNException(
					"Unknown algorithm for choosing initial centers.");
		}
	}

	/**
	 * Helper function
	 */
	void freeIndex() {
		if (root_)
			root_->~Node();
		root_ = NULL;
		pool_.free();
	}

	/**
	 * Computes the statistics of a node (mean, radius, variance).
	 *
	 * Params:
	 *     node = the node to use
	 *     indices = the indices of the points belonging to the node
	 */
	void computeNodeStatistics(NodePtr node, const std::vector<int>& indices) {
		size_t size = indices.size();

		DistanceType* mean = new DistanceType[veclen_];
		memoryCounter_ += int(veclen_ * sizeof(DistanceType));
		memset(mean, 0, veclen_ * sizeof(DistanceType));

		for (size_t i = 0; i < size; ++i) {
			ElementType* vec = points_[indices[i]];
			for (size_t j = 0; j < veclen_; ++j) {
				mean[j] += vec[j];
			}
		}
		DistanceType div_factor = DistanceType(1) / size;
		for (size_t j = 0; j < veclen_; ++j) {
			mean[j] *= div_factor;
		}

		DistanceType radius = 0;
		DistanceType average = 0;
		for (size_t i = 0; i < size; ++i) {
			DistanceType dist = distance_(mean, points_[indices[i]], veclen_);
			dist = std::pow(dist, 0.5f);
			if (dist > radius) {
				radius = dist;
			}
			average += dist;
		}
		average /= size;

		DistanceType variance = 0;
		for (size_t i = 0; i < size; ++i) {
			DistanceType dist = distance_(mean, points_[indices[i]], veclen_);
			dist = std::pow(dist, 0.5f);
			dist -= average;
			variance += dist * dist;
		}
		variance /= size;
		variance = std::pow(variance, 0.5f);

		node->average = average;
		node->variance = variance;
		node->radius = radius;
		node->pivot = mean;
	}

	/**
	 * The method responsible with actually doing the recursive hierarchical
	 * clustering
	 *
	 * Params:
	 *     node = the node to cluster
	 *     indices = indices of the points belonging to the current node
	 *     branching = the branching factor to use in the clustering
	 *
	 */
	void computeClustering(NodePtr node, int* indices, int indices_length,
			int branching) {
		node->size = indices_length;

		if (indices_length < branching) {
			node->points.resize(indices_length);
			for (int i = 0; i < indices_length; ++i) {
				node->points[i].index = indices[i];
				node->points[i].point = points_[indices[i]];
			}
			node->childs.clear();
			return;
		}

		std::vector<int> centers_idx(branching);
		int centers_length;
		(*chooseCenters_)(branching, indices, indices_length, &centers_idx[0],
				centers_length);

		if (centers_length < branching) {
			node->points.resize(indices_length);
			for (int i = 0; i < indices_length; ++i) {
				node->points[i].index = indices[i];
				node->points[i].point = points_[indices[i]];
			}
			node->childs.clear();
			return;
		}

		Matrix<double> dcenters(new double[branching * veclen_], branching,
				veclen_);
		for (int i = 0; i < centers_length; ++i) {
			ElementType* vec = points_[centers_idx[i]];
			for (size_t k = 0; k < veclen_; ++k) {
				dcenters[i][k] = double(vec[k]);
			}
		}

		std::vector<DistanceType> radiuses(branching, 0);
		std::vector<int> count(branching, 0);

		//	assign points to clusters
		std::vector<int> belongs_to(indices_length);
		for (int i = 0; i < indices_length; ++i) {

			DistanceType sq_dist = distance_(points_[indices[i]], dcenters[0],
					veclen_);
			belongs_to[i] = 0;
			for (int j = 1; j < branching; ++j) {
				DistanceType new_sq_dist = distance_(points_[indices[i]],
						dcenters[j], veclen_);
				if (sq_dist > new_sq_dist) {
					belongs_to[i] = j;
					sq_dist = new_sq_dist;
				}
			}
			if (sq_dist > radiuses[belongs_to[i]]) {
				radiuses[belongs_to[i]] = sq_dist;
			}
			count[belongs_to[i]]++;
		}

		bool converged = false;
		int iteration = 0;
		while (!converged && iteration < iterations_) {
			converged = true;
			iteration++;

			// compute the new cluster centers
			for (int i = 0; i < branching; ++i) {
				memset(dcenters[i], 0, sizeof(double) * veclen_);
				radiuses[i] = 0;
			}
			for (int i = 0; i < indices_length; ++i) {
				ElementType* vec = points_[indices[i]];
				double* center = dcenters[belongs_to[i]];
				for (size_t k = 0; k < veclen_; ++k) {
					center[k] += vec[k];
				}
			}
			for (int i = 0; i < branching; ++i) {
				int cnt = count[i];
				double div_factor = 1.0 / cnt;
				for (size_t k = 0; k < veclen_; ++k) {
					dcenters[i][k] *= div_factor;
				}
			}

			// reassign points to clusters
			for (int i = 0; i < indices_length; ++i) {
				DistanceType sq_dist = distance_(points_[indices[i]],
						dcenters[0], veclen_);
				int new_centroid = 0;
				for (int j = 1; j < branching; ++j) {
					DistanceType new_sq_dist = distance_(points_[indices[i]],
							dcenters[j], veclen_);
					if (sq_dist > new_sq_dist) {
						new_centroid = j;
						sq_dist = new_sq_dist;
					}
				}
				if (sq_dist > radiuses[new_centroid]) {
					radiuses[new_centroid] = sq_dist;
				}
				if (new_centroid != belongs_to[i]) {
					count[belongs_to[i]]--;
					count[new_centroid]++;
					belongs_to[i] = new_centroid;

					converged = false;
				}
			}

			for (int i = 0; i < branching; ++i) {
				// if one cluster converges to an empty cluster,
				// move an element into that cluster
				if (count[i] == 0) {
					int j = (i + 1) % branching;
					while (count[j] <= 1) {
						j = (j + 1) % branching;
					}

					for (int k = 0; k < indices_length; ++k) {
						if (belongs_to[k] == j) {
							belongs_to[k] = i;
							count[j]--;
							count[i]++;
							break;
						}
					}
					converged = false;
				}
			}

		}

		std::vector<DistanceType*> centers(branching);

		for (int i = 0; i < branching; ++i) {
			centers[i] = new DistanceType[veclen_];
			memoryCounter_ += veclen_ * sizeof(DistanceType);
			for (size_t k = 0; k < veclen_; ++k) {
				centers[i][k] = (DistanceType) dcenters[i][k];
			}
		}

		// compute kmeans clustering for each of the resulting clusters
		node->childs.resize(branching);
		int start = 0;
		int end = start;
		for (int c = 0; c < branching; ++c) {
			int s = count[c];

			DistanceType average = 0;
			for (int i = 0; i < indices_length; ++i) {
				if (belongs_to[i] == c) {
					float dist = distance_(centers[c], points_[indices[i]],
							veclen_);
					float tmp = std::pow(dist, 0.5f);
					average += tmp;
					std::swap(indices[i], indices[end]);
					std::swap(belongs_to[i], belongs_to[end]);
					end++;
				}
			}
			average /= s;

			DistanceType variance = 0;
			for (int i = 0; i < indices_length; ++i) {
				if (belongs_to[i] == c) {
					float dist = distance_(centers[c], points_[indices[i]],
							veclen_);
					float tmp = std::pow(dist, 0.5f);
					tmp -= average;
					variance += tmp * tmp;
				}
			}
			variance /= s;
			variance = std::pow(variance, 0.5f);

			node->childs[c] = new (pool_) Node();
			node->childs[c]->pivot = centers[c];
			node->childs[c]->radius = std::pow(radiuses[c], 0.5f);
//			node->childs[c]->radius = radiuses[c];
			node->childs[c]->average = average;
			node->childs[c]->variance = variance;

			computeClustering(node->childs[c], indices + start, end - start,
					end - start + 10);
//					branching);

			start = end;
		}

		delete[] dcenters.ptr();
	}

private:
	/** The branching factor used in the hierarchical k-means clustering */
	int branching_;

	/** Maximum number of iterations to use when performing k-means clustering */
	int iterations_;

	/** Algorithm for choosing the cluster centers */
	flann_centers_init_t centers_init_;

	/**
	 * Cluster border index. This is used in the tree search phase when determining
	 * the closest cluster to explore next. A zero value takes into account only
	 * the cluster centres, a value greater then zero also take into account the size
	 * of the cluster.
	 */
	float cb_index_;

	/**
	 * The root node in the tree.
	 */
	NodePtr root_;

	/**
	 * Pooled memory allocator.
	 */
	PooledAllocator pool_;

	/**
	 * Memory occupied by the index.
	 */
	int memoryCounter_;

	/**
	 * Algorithm used to choose initial centers
	 */
	CenterChooser<Distance>* chooseCenters_;

	USING_BASECLASS_SYMBOLS};

#endif //FLANN_KMEANS_H_
