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

using namespace std;

class Point
{
private:
	int id_point, id_cluster;
	vector<double> values;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "")
	{
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
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
		return total_values;
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

		int total_values = point.getTotalValues();

		for(int i = 0; i < total_values; i++)
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
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	// return ID of nearest center (uses euclidean distance)
	int getIDNearestCenter(Point point)
	{
		double sum = 0.0, min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
		{
			sum += pow(clusters[0].getCentralValue(i) -
					   point.getValue(i), 2.0);
		}
		
		// Sqrt potentially not necessary?
		min_dist = sqrt(sum);
		// min_dist = sum;

		for(int i = 1; i < K; i++)
		{
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
			{
				sum += pow(clusters[i].getCentralValue(j) -
						   point.getValue(j), 2.0);
			}

			dist = sqrt(sum);

			if(dist < min_dist)
			{
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations)
	{
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

	void run(vector<Point> & points)
	{
        auto begin = chrono::high_resolution_clock::now();

		if(K > total_points)
			return;

		// Manually initialize K cluster centroids with predefined values
		vector<vector<double>> initial_centroids = {
			{38609.2, 728.876, 264.767, 186.818, 1.42818, 0.689093, 39038, 221.602, 0.758159, 0.989029, 0.914007, 0.840847, 0.00686726, 0.00213791, 0.710003, 0.996999},
			{174059, 1588.67, 594.556, 374.795, 1.58763, 0.771401, 176399, 469.773, 0.776568, 0.986869, 0.864111, 0.792037, 0.00343789, 0.000840289, 0.628237, 0.991797},
			{47345.5, 829.504, 315.348, 193.204, 1.65068, 0.771733, 47935.3, 245.42, 0.740671, 0.987739, 0.867341, 0.784174, 0.00665986, 0.00157933, 0.619003, 0.995124},
			{71557, 1046.36, 392.086, 234.568, 1.67982, 0.795097, 72716.1, 301.742, 0.751715, 0.984066, 0.822499, 0.771608, 0.00548663, 0.00120628, 0.596954, 0.992484},
			{85686, 1141.11, 429.742, 255.997, 1.68471, 0.798436, 87043.1, 330.051, 0.755788, 0.984409, 0.827205, 0.769505, 0.00503001, 0.00109253, 0.593336, 0.99205},
			{58464.4, 953.494, 371.873, 202.469, 1.85603, 0.829191, 59350.5, 272.711, 0.723485, 0.985113, 0.809081, 0.736804, 0.00637706, 0.00117043, 0.545769, 0.992736},
			{29664.5, 639.033, 234.918, 161.141, 1.46392, 0.7195, 30023.8, 194.083, 0.755489, 0.987979, 0.911338, 0.82758, 0.00796609, 0.00231207, 0.686245, 0.996982}
		};

		// Assign initial centroids manually instead of randomly selecting points
		for (int i = 0; i < K; i++) {
			Cluster cluster(i, initial_centroids[i]);
			clusters.push_back(cluster);
		}

        auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;

		while(true)
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
				for(int j = 0; j < total_values; j++)
				{
					int total_points_cluster = clusters[i].getTotalPoints();
					double sum = 0.0;

					if(total_points_cluster > 0)
					{
						for(int p = 0; p < total_points_cluster; p++)
							sum += clusters[i].getPoint(p).getValue(j);
						clusters[i].setCentralValue(j, sum / total_points_cluster);
					}
				}
			}

			if(done == true || iter >= max_iterations)
			{
				cout << "Break in iteration " << iter << "\n\n";
				break;
			}

			iter++;
		}
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
				// for(int p = 0; p < total_values; p++)
				// 	cout << clusters[i].getPoint(j).getValue(p) << " ";
				// string point_name = clusters[i].getPoint(j).getName();
				// if(point_name != "")
				// 	cout << "- " << point_name;
			}

			cout << "Cluster values: ";

			for(int j = 0; j < total_values; j++)
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
	int total_points, total_values, K, max_iterations, has_name;
	ss >> total_points >> total_values >> K >> max_iterations >> has_name;

	// Print all the inputed values
	cout << "total_points = " << total_points << endl;
	cout << "total_values = " << total_values << endl;
	cout << "K = " << K << endl;
	cout << "max_iterations = " << max_iterations << endl;
	cout << "has_name = " << has_name << endl;

	if (total_points == 0 || total_values == 0 || K == 0 || max_iterations == 0)
	{
		cout << "Invalid input" << endl;
		return 1;
	}

	vector<Point> points;
	string point_name;

	for(int i = 0; i < total_points; i++)
	{
		vector<double> values;

		for(int j = 0; j < total_values; j++)
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

	KMeans kmeans(K, total_points, total_values, max_iterations);
	kmeans.run(points);

	return 0;
}
