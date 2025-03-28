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
#include <immintrin.h>  // AVX2 intrinsics, using for low-level SIMD operations

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
	vector<double> central_values;     // K * total_attr
	vector<double> attribute_sums;     // K * total_attr
	vector<int>    cluster_counts;     // K

	// Helper function to get index in flattened vectors
	int getClusterIndex(int cluster_id, int attr) {
		return cluster_id * total_attr + attr;
	}

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		// Calculate distance to first cluster
		for(int i = 0; i < total_attr; i++)
		{
			double diff = central_values[i] - point.getValue(i);
			sum += diff * diff;
		}
		min_dist = sum;

		double* p_vals = point.getValues().data();
		for(int i = 1; i < K; i++)
		{
			sum = 0.0;
			double* c_vals = &central_values[getClusterIndex(i, 0)];

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
		central_values.resize(K * total_attr);
		attribute_sums.resize(K * total_attr);
		cluster_counts.resize(K);
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
					cluster_counts[i] = 1;
					
					// Copy point values to central values
					for(int j = 0; j < total_attr; j++) {
						central_values[getClusterIndex(i, j)] = points[index_point].getValue(j);
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


		// ======================= RUN KMEANS ========================= //
		// 4. Turn while(true) into a for loop (still w/ break statement for stopping condition)
		int iter = 1;
		bool done = false;
		for (; !done && iter <= max_iterations; iter++)
		{
			done = true;

			// Clear attribute sums at start of iteration
			fill(attribute_sums.begin(), attribute_sums.end(), 0.0);

			// Associate each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						cluster_counts[id_old_cluster]--;

					points[i].setCluster(id_nearest_center);
					cluster_counts[id_nearest_center]++;
					done = false;
				}
 
				// Add point values to attribute sums
				double* p_vals = points[i].getValues().data();
				double* sums = &attribute_sums[getClusterIndex(id_nearest_center, 0)];
				#pragma omp simd // Trying OpenMP pragma to see if it helps
				for(int j = 0; j < total_attr; j++) {
					sums[j] += p_vals[j];
				}
			}

			// Recalculate the center of each cluster
			for(int i = 0; i < K; i++)
			{
				if(cluster_counts[i] > 0) {
					double* cent_vals = &central_values[getClusterIndex(i, 0)];
					double* sums = &attribute_sums[getClusterIndex(i, 0)];
					#pragma omp simd // Trying OpenMP pragma to see if it helps
					for(int j = 0; j < total_attr; j++) {
						cent_vals[j] = sums[j] / cluster_counts[i];
					}
				}
			}
		}
		cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();


		// Output Results
		for(int i = 0; i < K; i++)
		{
			cout << "Cluster " << i + 1 << ": ";
			for(int j = 0; j < total_attr; j++)
				cout << central_values[getClusterIndex(i, j)] << " ";
			cout << "\n\n";
		}
        cout << "TOTAL EXECUTION TIME = "<<chrono::duration_cast<chrono::microseconds>(end-begin).count()<<"μs\n";
        cout << "TIME PHASE 1 = "<<chrono::duration_cast<chrono::microseconds>(end_phase1-begin).count()<<"μs\n";
        cout << "TIME PHASE 2 = "<<chrono::duration_cast<chrono::microseconds>(end-end_phase1).count()<<"μs\n\n\n" << endl;
	}
};

int main(int argc, char *argv[])
{
	srand (123); // Set seed for reproducibility

	string first_line;
	getline(cin, first_line);

	// IMPORTANT: Remove BOM if it exists
	if (first_line.size() > 0 && first_line[0] == '\xEF' && first_line[1] == '\xBB' && first_line[2] == '\xBF') {
		first_line.erase(0, 3);
	}

	// Print dataset info
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

	// Read in points from dataset
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
	}
	KMeans kmeans(K, total_points, total_attr, max_iterations);
	kmeans.run(points);
	return 0;
}
