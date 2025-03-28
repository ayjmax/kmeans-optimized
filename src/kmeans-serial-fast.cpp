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
	unordered_map<int, Point> points; // 6. Make removing a Point O(1) operation by using a map; 11. change map to unordered_map to improve performance
	vector<double> attributeSums; // 5. Add a vector to store the sum of all attributes of all points in the cluster; 12. make this a unique_ptr (revoked)


public:
	// Constructor to initialize with a random existing point
	Cluster(int id_cluster, Point point, int total_attr)
	{
		this->id_cluster = id_cluster;
		this->total_attr = total_attr;

		for(int i = 0; i < total_attr; i++)
			central_values.push_back(point.getValue(i));

		points.insert(pair<int, Point>(point.getID(), point));

		// 5. Assign initial 16 values of 0.0 to attributeSums
		this->attributeSums.assign(total_attr, 0.0);
	}

	// New constructor to initialize with predefined central values
	Cluster (int id_cluster, vector<double>& central_values, int total_attr)
	{
		this->id_cluster = id_cluster;
		this->central_values = central_values;
		this->total_attr = total_attr;
		// 5. Assign initial 16 values of 0.0 to attributeSums
		this->attributeSums.assign(total_attr, 0.0);

		// 12. Allocate memory for attributeSums (revoked)
		// attributeSums = std::make_unique<double[]>(total_attr);
		// std::fill(attributeSums.get(), attributeSums.get() + total_attr, 0.0);
	}

	void addPoint(Point point)
	{
		points.insert(pair<int, Point>(point.getID(), point));
	}

	// 6. Remove a Point in O(1) time
	bool removePoint(int id_point)
	{
		points.erase(id_point); 
		return true;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	// 7. Getter for central values (centroid)
	vector<double>& getCentralValues()
	{
		return central_values;
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

    // 6. Getter for all points in the cluster
    vector<Point> getAllPoints()
    {
        vector<Point> allPoints;
        for (const auto& pair : points)
        {
            allPoints.push_back(pair.second);
        }
        return allPoints;
    }

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}

	// 5. Update the central values based on attributeSums
	void updateCentralValues() {
		int total_points = getTotalPoints();
		if (total_points > 0) { // Safety check
			for(int i = 0; i < total_attr; i++) // 16 attributes
			{
				central_values[i] = attributeSums[i] / total_points;
			}
		}
	}

	// 5. Add the attribute values of the point to attributeSums
	void addAttributeSums(Point point)
	{
		for(int i = 0; i < total_attr; i++)
		{
			attributeSums[i] += point.getValue(i);
		}
	}

	// 5. Clear attributeSums
	void clearAttributeSums()
	{
		// replaces all 16 values w/ 0.0
		fill(attributeSums.begin(), attributeSums.end(), 0.0);

		// 12. Clear attributeSums
		// fill(attributeSums.get(), attributeSums.get() + total_attr, 0.0);
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_attr, total_points, max_iterations;
	// vector<unique_ptr<Cluster>> clusters;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;
		// vector<double> euclideanDistanceVals(total_attr, 0.0); // 3. Try putting results of calculations into a vector then sum over the vector
		// vector<double> sums(K, 0.0);

		for(int i = 0; i < total_attr; i++)
		{
			double diff = clusters[0].getCentralValue(i) - point.getValue(i);
			sum += diff * diff; // 10. Replace pow with multiplication
		}

		// 1. Sqrt potentially not necessary?
		min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			sum = 0.0;
			const vector<double>& pointValues = point.getValues();
			const vector<double>& clusterValues = clusters[i].getCentralValues();
			for(int j = 0; j < total_attr; j++)
			{
				// euclideanDistanceVals[j] = pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
				double diff = clusters[i].getCentralValue(j) - point.getValue(j);
				sum += diff * diff; // 10. Replace pow with multiplication
			}

			// sum = accumulate(euclideanDistanceVals.begin(), euclideanDistanceVals.end(), 0.0);
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

	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();

		if(K > total_points)
			return;

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

        auto end_phase1 = chrono::high_resolution_clock::now();


		// ############################# RUN KMEANS ############################## //
		// 4. Turn while(true) into a for loop (still w/ break statement for stopping condition)
		int iter = 1;
		bool done = false;
		for (; !done && iter <= max_iterations; iter++)
		{
			done = true;
			// 5. Clear the sum of all attributes of all points in the cluster
			for(int i = 0; i < K; i++) // 7 clusters
			{
				clusters[i].clearAttributeSums();
			}

			// Associate each point to the nearest center
			for(int i = 0; i < total_points; i++)
			{
				int id_old_cluster = points[i].getCluster();
				int id_nearest_center = getIDNearestCenter(points[i]);

				if(id_old_cluster != id_nearest_center)
				{
					if(id_old_cluster != -1)
						clusters[id_old_cluster].removePoint(points[i].getID());

					points[i].setCluster(id_nearest_center);
					clusters[id_nearest_center].addPoint(points[i]);
					done = false;
				}
 
				// 5. Add the attributes of the point to the sum of all attributes of all points in the cluster
				clusters[id_nearest_center].addAttributeSums(points[i]);
			}

			// Recalculate the center of each cluster
			for(int i = 0; i < K; i++)
			{
				// 5. Calculate the new centroid based on attributeSums
				clusters[i].updateCentralValues();
			}
		}
		cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();


		// Output Results
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "Cluster " << clusters[i].getID() + 1 << " values: ";
			for(int j = 0; j < total_attr; j++)
				cout << clusters[i].getCentralValue(j) << " ";
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
