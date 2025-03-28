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

class Cluster
{
private:
	int id_cluster;
	int total_attr;
	vector<double> central_values;
	vector<double> attributeSums; // S5. Add a vector to store the sum of all attributes of all points in the cluster; 12. make this a unique_ptr
	int num_points;

public:
	// Constructor to initialize with a random existing point
	Cluster(int id_cluster, Point point, int total_attr)
	{
		this->id_cluster = id_cluster;
		this->total_attr = total_attr;
		this->num_points = 1; // NOTE: When using this constructor, one point is already in the cluster
		this->attributeSums.assign(total_attr, 0.0);

		for(int i = 0; i < total_attr; i++)
			central_values.push_back(point.getValue(i));
	}

	// New constructor to initialize with predefined central values
	Cluster (int id_cluster, vector<double>& central_values, int total_attr)
	{
		this->id_cluster = id_cluster;
		this->central_values = central_values;
		this->total_attr = total_attr;
		this->attributeSums.assign(total_attr, 0.0);
		this->num_points = 0;
	}

	// P3
	bool setNumPoints(int num_points) {
		this->num_points = num_points;
		return true;
	}

	// P3
	void addToNumPoints(int points_diff) {
		num_points += points_diff;
	}

	// P3
	int getNumPoints() {
		return num_points;
	}

	void incrementNumPoints() {
		num_points++;
	}

	void decrementNumPoints() {
		num_points--;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	// S7. Getter for central values (centroid)
	vector<double>& getCentralValues()
	{
		return central_values;
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	int getTotalPoints()
	{
		return num_points;
	}

	int getID()
	{
		return id_cluster;
	}

	// S5. Update the central values based on attributeSums
	void updateCentralValues() {
		if (num_points <= 0) return;
		for(int i = 0; i < total_attr; i++) // 16 attributes
		{
			central_values[i] = attributeSums[i] / num_points;
		}
	}

	// S5. Add the attribute values of the point to attributeSums
	void addAttributeSums(Point point)
	{
		for(int i = 0; i < total_attr; i++)
		{
			attributeSums[i] += point.getValue(i);
		}
	}

	void addAttributeSums(vector<double> attributeSums)
	{
		for(int i = 0; i < total_attr; i++) {
			this->attributeSums[i] += attributeSums[i];
		}
	}

	// S5. Clear attributeSums
	void clearAttributeSums()
	{
		fill(attributeSums.begin(), attributeSums.end(), 0.0);
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_attr, total_points, max_iterations;
	vector<Cluster> clusters;

	// Return ID of nearest center (uses euclidean distance)
	int findNearestCluster(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_attr; i++)
		{
			double diff = clusters[0].getCentralValue(i) - point.getValue(i);
			sum += diff * diff;
		}

		min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			sum = 0.0;
			for(int j = 0; j < total_attr; j++)
			{
				double diff = clusters[i].getCentralValue(j) - point.getValue(j);
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
					Cluster cluster(i, points[index_point], total_attr);
					clusters.push_back(cluster);
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
				for (int j = 0; j < total_attr; j++) {
					local_sums[id_nearest_center][j] += points[i].getValue(j);
				}
			});

			// P3. Updating num_points using the values of the differences accumulated in each threadLocalPointDiffs
			for (const auto& local_diffs : thread_local_point_diffs) {
				for (int i = 0; i < K; i++) {
					if (done && local_diffs[i] != 0) { // Moved 'done' check here to remove race condition/contention
						done = false;
					}
					clusters[i].addToNumPoints(local_diffs[i]);
				}
			}

			// P3
			for (const auto& local_sums : thread_local_attribute_sums) {
				for (int i = 0; i < K; i++) {
					clusters[i].addAttributeSums(local_sums[i]);
				}
			}

			// P2. parallelize clearing attributeSums
			tbb::parallel_for(0, K, 1, [&](int i) {
				clusters[i].updateCentralValues();
				clusters[i].clearAttributeSums();
			});
		}

		cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();

		// Output Results
		for(int i = 0; i < K; i++)
		{
			cout << "Cluster " << clusters[i].getID() + 1 << ": ";
			for(int j = 0; j < total_attr; j++)
				cout << clusters[i].getCentralValue(j) << " ";
			cout << "\n\n";
		}
		cout << "TOTAL EXECUTION TIME = "<<chrono::duration_cast<chrono::microseconds>(end-begin).count()<<"μs\n";
		cout << "TIME PHASE 1 = "<<chrono::duration_cast<chrono::microseconds>(end_phase1-begin).count()<<"μs\n";
		cout << "TIME PHASE 2 = "<<chrono::duration_cast<chrono::microseconds>(end-end_phase1).count()<<"μs\n" << endl;
		cout << "AV TIME PER ITERATION = " << (chrono::duration_cast<chrono::microseconds>(end-begin).count() / iter) << "μs\n\n\n" << endl;
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

		// Clear any remaining values in the line
		cin.ignore(numeric_limits<streamsize>::max(), '\n');
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


