// Implementation of the KMeans Algorithm
// reference: https://github.com/marcoscastro/kmeans

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <sstream> // Include the sstream header for stringstream
#include <climits>
#include <numeric>
#include <unordered_map>
#include <memory> // for std::unique_ptr
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <mutex>
#include <tbb/global_control.h> // to control the number of threads

using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_attr;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_attr = values.size();

		for(int i = 0; i < total_attr; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

	int getID()
	{
		return id_point;
	}

	void setCluster(int id_cluster)
	{
		this->id_cluster = id_cluster;
	}

	int getCluster()
	{
		return id_cluster;
	}

	double getValue(int index)
	{
		return values[index];
	}

	vector<double>& getValues()
	{
		return values;
	}

	int getTotalValues()
	{
		return total_attr;
	}

	void addValue(double value)
	{
		values.push_back(value);
	}

	string getName()
	{
		return name;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_attr, total_points, max_iterations;
	vector<double> centralValues;     // K * total_attr
	vector<double> attributeSums;     // K * total_attr
	vector<int>    clusterCounts;     // K

	// Helper function to get index in flattened vectors
	int getClusterIndex(int cluster_id, int attr) {
		return cluster_id * total_attr + attr;
	}

	// Return ID of nearest center (uses euclidean distance)
	int findNearestCluster(Point point)
	{
		double sum = 0.0, min_dist;
 		int id_cluster_center = 0;
 		for(int i = 0; i < total_attr; i++)
		{
			double diff = centralValues[i] - point.getValue(i);
			sum += diff * diff;
		}

		min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			sum = 0.0;
			double* c_vals = &centralValues[getClusterIndex(i, 0)];
			double* p_vals = point.getValues().data();
 
			#pragma omp simd
			for(int j = 0; j < total_attr; j++)
			{
				double diff = c_vals[j] - p_vals[j];
				sum += diff * diff;
			}
 
			if (sum < min_dist)
			{
				min_dist = sum;
				id_cluster_center = i;
			}
		}
		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_attr, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_attr = total_attr;
		this->max_iterations = max_iterations;
		
		// Initialize vectors with correct sizes
		centralValues.resize(K * total_attr);
		attributeSums.resize(K * total_attr);
		clusterCounts.resize(K);
	}

	void initializeClusterCentroids(vector<Point> & points)
	{
		// Manually initialize K cluster centroids with unique, random points
		vector<int> prohibited_indexes;
		for(int i = 0; i < K; i++)
		{
			while(true)
			{
				int index_point = rand() % total_points; // Random seed is defined in main

				if(find(prohibited_indexes.begin(), prohibited_indexes.end(),
						index_point) == prohibited_indexes.end())
				{
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					clusterCounts[i] = 1;
					
					// Copy point values to central values
					for(int j = 0; j < total_attr; j++) {
						centralValues[getClusterIndex(i, j)] = points[index_point].getValue(j);
					}
					break;
				}
			}
		}
		return;
	}

	void run(vector<Point> & points)
	{
		if(K > total_points)
			return;

        auto begin = chrono::high_resolution_clock::now();
		initializeClusterCentroids(points);
        auto end_phase1 = chrono::high_resolution_clock::now();


		// ======================= RUN KMEANS ======================= //
		int iter = 1;
		bool done = false;
		for (; !done && iter <= max_iterations; iter++)
		{
			done = true;
			tbb::enumerable_thread_specific<vector<int>> thread_local_point_diffs(
				[&]() { return vector<int>(K, 0); }
			); // Basically, this creates a vector of size K with all elements initialized to 0 per thread
			tbb::enumerable_thread_specific<vector<vector<double>>> thread_local_attribute_sums(
				[&]() { return vector<vector<double>>(K, vector<double>(total_attr, 0.0)); }
			); // 2-D vector, K x total_attr: Rows are clusters, columns are attributes

			// P1. Parallel for over all points to assign them to the nearest cluster
			tbb::parallel_for(0, total_points, 1, [&](int i) {
				// NOTE: Due to the nature of findNearestCluster, cluster information should NOT be changed in this loop
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = findNearestCluster(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					// P3
					auto& local_diffs = thread_local_point_diffs.local();
					if (id_old_cluster != -1) {
						local_diffs[id_old_cluster]--;
					}
					local_diffs[id_nearest_center]++;

					points[i].setCluster(id_nearest_center);
				}

				// P3
				auto& local_sums = thread_local_attribute_sums.local();
				double* p_vals = points[i].getValues().data();
				#pragma omp simd
				for (int j = 0; j < total_attr; j++) {
					local_sums[id_nearest_center][j] += p_vals[j];
				}
			});

			// P3. Updating num_points using the values of the differences accumulated in each threadLocalPointDiffs
			for (const auto& local_diffs : thread_local_point_diffs) {
				for (int i = 0; i < K; i++) {
					if (done && local_diffs[i] != 0) { // Moved 'done' check here to remove race condition/contention
						done = false;
					}
					clusterCounts[i] += local_diffs[i];
				}
			}

			// P3. Update attribute sums
			for (const auto& local_sums : thread_local_attribute_sums) {
				for (int i = 0; i < K; i++) {
					double* sums = &attributeSums[getClusterIndex(i, 0)];
					#pragma omp simd
					for (int j = 0; j < total_attr; j++) {
						sums[j] += local_sums[i][j];
					}
				}
			}

			// P2. parallelize clearing attributeSums
			tbb::parallel_for(0, K, 1, [&](int i) {
				if(clusterCounts[i] > 0) {
					double* cent_vals = &centralValues[getClusterIndex(i, 0)];
					double* sums = &attributeSums[getClusterIndex(i, 0)];
					#pragma omp simd
					for(int j = 0; j < total_attr; j++) {
						cent_vals[j] = sums[j] / clusterCounts[i];
					}
				}
				// Clear attribute sums for next iteration
				fill(&attributeSums[getClusterIndex(i, 0)], &attributeSums[getClusterIndex(i, total_attr)], 0.0);
			});
		}

		cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();

		// Output Results
		for(int i = 0; i < K; i++)
		{
			cout << "Cluster " << i + 1 << ": ";
			for(int j = 0; j < total_attr; j++)
				cout << centralValues[getClusterIndex(i, j)] << " ";
			cout << "\n\n";
		}
		cout << "TOTAL EXECUTION TIME = "<<chrono::duration_cast<chrono::microseconds>(end-begin).count()<<"μs\n";
		cout << "TIME PHASE 1 = "<<chrono::duration_cast<chrono::microseconds>(end_phase1-begin).count()<<"μs\n";
		cout << "TIME PHASE 2 = "<<chrono::duration_cast<chrono::microseconds>(end-end_phase1).count()<<"μs\n\n\n" << endl;
	}
};

int main(int argc, char *argv[])
{
	string first_line;
	getline(cin, first_line);

	// IMPORTANT: Remove byte-order-mark (BOM) if it exists
	if (first_line.size() > 0 && first_line[0] == '\xEF' && first_line[1] == '\xBB' && first_line[2] == '\xBF') {
		first_line.erase(0, 3);
	}

	cout << "Dataset info: " << first_line << endl;

	// Use stringstream to split the first line into integers
	stringstream ss(first_line);
	int total_points, total_attr, K, max_iterations, has_name;
	ss >> total_points >> total_attr >> K >> max_iterations >> has_name;

	if (total_points == 0 || total_attr == 0 || K == 0 || max_iterations == 0)
	{
		cout << "Invalid input" << endl;
		return 1;
	}



	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_attr; j++)
		{
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name)
		{
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else
		{
			Point p(i, values);
			points.push_back(p);
		}
	}

	vector<Point> backup_points = points; // make a backup copy

	// for (int threads : {1, 2, 4, 8, 16, 32, 50, 100, 500}) {
	// 	tbb::global_control c(tbb::global_control::max_allowed_parallelism, threads);
		srand (123); // For reproducibility

		points = backup_points; // restore the backup copy

		// cout << "Threads: " << threads << endl;
		KMeans kmeans(K, total_points, total_attr, max_iterations);
		kmeans.run(points);
	// }

	return 0;
}


