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
	vector<double> central_values;
	vector<Point> points;

public:
	// Constructor to initialize with a random existing point
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_attr = point.getTotalValues();

		for(int i = 0; i < total_attr; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

	// New constructor to initialize with predefined central values
	Cluster (int id_cluster, vector<double>& central_values)
	{
		this->id_cluster = id_cluster;
		this->central_values = central_values;
	}

	void addPoint(Point point)
	{
		points.push_back(point);
	}

	bool removePoint(int id_point)
	{
		int total_points = points.size();

		for(int i = 0; i < total_points; i++)
		{
			if(points[i].getID() == id_point)
			{
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index)
	{
		return central_values[index];
	}

	void setCentralValue(int index, double value)
	{
		central_values[index] = value;
	}

	Point getPoint(int index)
	{
		return points[index];
	}

	int getTotalPoints()
	{
		return points.size();
	}

	int getID()
	{
		return id_cluster;
	}
};

class KMeans
{
private:
	int K; // number of clusters
	int total_attr, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;
		vector<double> euclideanDistanceVals(total_attr, 0.0); // 3. Try putting results of calculations into a vector then sum over the vector

		for(int i = 0; i < total_attr; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}

		// min_dist = sqrt(sum); // 1. Sqrt potentially not necessary?
		min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			for(int j = 0; j < total_attr; j++)
			{
				euclideanDistanceVals[j] = pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
			}

			// dist = sqrt(sum); // 1. Sqrt potentially not necessary?
			sum = accumulate(euclideanDistanceVals.begin(), euclideanDistanceVals.end(), 0.0); // 3. Try putting results of calculations into a vector then sum over the vector

			if(sum < min_dist)
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

		// Manually initialize K cluster centroids with predefined values
		vector<vector<double>> initial_centroids = {
			{43266, 772.889, 287.013, 193.007, 1.48706, 0.740125, 43691, 234.708, 0.77649, 0.990273, 0.91017, 0.817763, 0.00663368, 0.00182997, 0.668736, 0.994449},
			{77590, 1074.37, 409.117, 245.292, 1.66788, 0.800326, 79050, 314.31, 0.798884, 0.981531, 0.844714, 0.768264, 0.0052728, 0.00113309, 0.59023, 0.984431},
			{75720, 1097.63, 390.076, 248.122, 1.57211, 0.771617, 77033, 310.499, 0.703848, 0.982955, 0.789781, 0.795997, 0.00515155, 0.00127575, 0.633612, 0.996108},
			{37064, 710.701, 265.11, 178.209, 1.48763, 0.740362, 37390, 217.236, 0.706142, 0.991281, 0.922122, 0.819417, 0.00715277, 0.00198918, 0.671444, 0.998863},
			{71882, 1040.31, 408.824, 226.343, 1.80621, 0.832753, 73208, 302.528, 0.723516, 0.981887, 0.834644, 0.739996, 0.00568743, 0.00105199, 0.547594, 0.989071},
			{52187, 975.927, 311.194, 215.771, 1.44224, 0.720587, 53664, 257.772, 0.709043, 0.972477, 0.688553, 0.828333, 0.00596305, 0.00173169, 0.686136, 0.989575},
			{80724, 1071.17, 416.301, 247.545, 1.68172, 0.804, 81330, 320.595, 0.799723, 0.992549, 0.884088, 0.770103, 0.00515709, 0.00111887, 0.593058, 0.997358}
		};

		// Assign initial centroids manually instead of randomly selecting points
		for (int i = 0; i < K; i++) {
			Cluster cluster(i, initial_centroids[i]);
			clusters.push_back(cluster);
		}


        auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;

		// 4. Turn while(true) into a for loop (still w/ break statement for stopping condition)
		for (int iter = 1; iter <= max_iterations; iter++)
		{
			bool done = true;

			// associates each point to the nearest center
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
			}

			// recalculating the center of each cluster
			for(int i = 0; i < K; i++)
			{
				for(int j = 0; j < total_attr; j++) // loop through all attributes/fields
				{
					int cluster_num_points = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(cluster_num_points > 0)
					{
						for(int p = 0; p < cluster_num_points; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / cluster_num_points);
					}
				}
			}

			if(done == true) // Keep break condition
			{
				break;
			}
		}
		cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();

		// shows elements of clusters
		for(int i = 0; i < K; i++)
		{
			int total_points_cluster =  clusters[i].getTotalPoints();

			cout << "############################################################# Cluster " << clusters[i].getID() + 1 << " ";
			cout << "#############################################################" << endl;
			for(int j = 0; j < total_points_cluster; j++)
			{
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << "-> " << clusters[i].getPoint(j).getName() << endl;
				// for(int p = 0; p < total_attr; p++)
				// 	cout << clusters[i].getPoint(j).getValue(p) << " ";
				// string point_name = clusters[i].getPoint(j).getName();
				// if(point_name != "")
				// 	cout << "- " << point_name;
			}

			cout << "Cluster values: ";

			for(int j = 0; j < total_attr; j++)
				cout << clusters[i].getCentralValue(j) << " ";
			
			cout << "\n\n\n" << endl;
		}

		cout << "\n\n";
		cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"μs\n";

		cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"μs\n";

		cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"μs\n";
	}
};

int main(int argc, char *argv[])
{
	srand (time(NULL));

	string first_line;
	getline(cin, first_line);

	// IMPORTANT: Remove BOM if it exists
	if (first_line.size() > 0 && first_line[0] == '\xEF' && first_line[1] == '\xBB' && first_line[2] == '\xBF') {
		first_line.erase(0, 3);
	}

	cout << "first_line = " << first_line << endl;

	// Use stringstream to split the first line into integers
	stringstream ss(first_line);
	int total_points, total_attr, K, max_iterations, has_name;
	ss >> total_points >> total_attr >> K >> max_iterations >> has_name;

	// Print all the inputed values
	cout << "total_points = " << total_points << endl;
	cout << "total_attr = " << total_attr << endl;
	cout << "K = " << K << endl;
	cout << "max_iterations = " << max_iterations << endl;
	cout << "has_name = " << has_name << endl;

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

	KMeans kmeans(K, total_points, total_attr, max_iterations);
	kmeans.run(points);

	return 0;
}
