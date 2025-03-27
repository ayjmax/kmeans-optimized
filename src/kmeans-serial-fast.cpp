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
	unordered_map<int, Point> points; // 6. Make removing a Point O(1) operation; 11. change map to unordered_map
	vector<double> attributeSums; // 5. Add a vector to store the sum of all attributes of all points in the cluster; 12. make this a unique_ptr

public:
	// Constructor to initialize with a random existing point
	Cluster(int id_cluster, Point point)
	{
		this->id_cluster = id_cluster;

		int total_attr = point.getTotalValues();

		for(int i = 0; i < total_attr; i++)
			central_values.push_back(point.getValue(i));

		points.insert(pair<int, Point>(point.getID(), point));
	}

	// New constructor to initialize with predefined central values
	Cluster (int id_cluster, vector<double>& central_values, int total_attr)
	{
		this->id_cluster = id_cluster;
		this->central_values = central_values;
		this->total_attr = total_attr;
		// 5. Assign initial 16 values of 0.0 to attributeSums
		this->attributeSums.assign(total_attr, 0.0);

		// 12. Allocate memory for attributeSums
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
		for(int i = 0; i < point.getTotalValues(); i++)
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
			Cluster cluster(i, initial_centroids[i], total_attr);
			clusters.push_back(cluster); // 12. Use make_unique
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
			cout << "############################################################# Cluster " << clusters[i].getID() + 1 << " ";
			cout << "#############################################################" << endl;

			vector<Point> allPoints =  clusters[i].getAllPoints();
			for(Point p : allPoints)
			{
				cout << "Point " << p.getID() << "-> " << p.getName() << '\n';
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
