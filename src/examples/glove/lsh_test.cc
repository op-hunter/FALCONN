/*
 * An example program that takes a GloVe
 * (http://nlp.stanford.edu/projects/glove/) dataset and builds a cross-polytope
 * LSH table with the following goal in mind: for a random subset of NUM_QUERIES
 * points, we would like to find a nearest neighbor (w.r.t. cosine similarity)
 * with probability at least 0.9.
 *
 * There is a function get_default_parameters, which you can use to set the
 * parameters automatically (in the code, we show how it could have been used).
 * However, we recommend to set parameters manually to maximize the performance.
 *
 * You need to specify:
 *   - NUM_HASH_TABLES, which affects the memory usage: the larger it is, the
 *     better (unless it's too large). Despite that, it's usually a good idea
 *     to start with say 10 tables, and then increase it gradually, while
 *     observing the effect it makes.
 *   - NUM_HASH_BITS, that controls the number of buckets per table,
 *     usually it should be around the binary logarithm of the number of data
 *     points
 *   - NUM_ROTATIONS, which controls the number of pseudo-random rotations for
 *     the cross-polytope LSH, set it to 1 for the dense data, and 2 for the
 *     sparse data (for GloVe we set it to 1)
 *
 * The code sets the number of probes automatically. Also, it recenters the
 * dataset for improved partitioning. Since after recentering vectors are not
 * unit anymore we should use the Euclidean distance in the data structure.
 */

#include <falconn/lsh_nn_table.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>
#include "/usr/include/hdf5/serial/hdf5.h"
#include "/usr/include/hdf5/serial/H5Cpp.h"
using namespace H5;

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::DenseVector;
using falconn::SparseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;

typedef DenseVector<float> Point;
//typedef SparseVector<float> Point;// indices default int32_t, should be sorted

const string FILE_PATH = "/home/zilliz/workspace/data/";
const string FILE_NAME = "sift-128-euclidean.hdf5";
const int NUM_QUERIES = 10;
const int SEED = 4057218;
const int NUM_HASH_TABLES = 50;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 1;

/*
 * An auxiliary function that reads raw data from hdf5 file that is pre-provided
 */
void LoadData(const std::string file_location, float *&data, const std::string data_name, int &dim, int &num_vets) {
  hid_t fd;
  herr_t status;
  fd = H5Fopen(file_location.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataset_id;
  dataset_id = H5Dopen2(fd, data_name.c_str(), H5P_DEFAULT);
  status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  hid_t dspace = H5Dget_space(dataset_id);
  hsize_t dims[2];
  H5Sget_simple_extent_dims(dspace, dims, NULL);
  num_vets = dims[0];
  dim = dims[1];
  status = H5Dclose(dataset_id);
  status = H5Fclose(fd);
}

void ReadData(float *&data, vector<Point> *dataset, const int dim, const int n_total) {
  float* pdata = data;
  dataset->clear();
  for (auto i = 0; i < n_total; ++ i) {
//    cout << "start i: " << i << ", p.size: " << p.size() << endl;
    Point p;
    p.resize(dim);
//    cout << "after resize, p.size: " << p.size() << endl;
    for (auto j = 0; j < dim; ++ j) {
      p[j] = *pdata;
      ++ pdata;
    }
//    cout << "i: " << i << ", offset: " << pdata - data << ", p.size: " << p.size() << endl;
    dataset->push_back(p);
  }
  std::cout << "ReadData Done! dataset size: " << dataset->size() << std::endl;
}

/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * Normalizes the dataset.
 */
void normalize(vector<Point> *dataset) {
  for (auto &p : *dataset) {
    p.normalize();
  }
}

/*
 * Chooses a random subset of the dataset to be the queries.
 */
void GenQueries(vector<Point> *dataset, vector<Point> *queries) {
  mt19937_64 gen(SEED);
  queries->clear();
  cout << "gen queries id: " << endl;
  for (int i = 0; i < NUM_QUERIES; ++i) {
    uniform_int_distribution<> u(0, dataset->size() - 1);
    int ind = u(gen);
    cout << "the " << i << "th query id is " << ind << endl;
    queries->push_back((*dataset)[ind]);
  }
}

/*
 * Chooses a random subset of the dataset to be the queries. The queries are
 * taken out of the dataset.
 */
void gen_queries(vector<Point> *dataset, vector<Point> *queries) {
  mt19937_64 gen(SEED);
  queries->clear();
  for (int i = 0; i < NUM_QUERIES; ++i) {
    uniform_int_distribution<> u(0, dataset->size() - 1);
    int ind = u(gen);
    queries->push_back((*dataset)[ind]);
    (*dataset)[ind] = dataset->back();
    dataset->pop_back();
  }
}

/*
 * Generates answers for the queries using the (optimized) linear scan. Brute Force
 */
void GenAnswers(const vector<Point> &dataset, const vector<Point> &queries,
                 vector<int> *answers) {
  answers->resize(queries.size());
  int outer_counter = 0;
  for (const auto &query : queries) {
    float best = -10.0;
    int inner_counter = 0;
    for (const auto &datapoint : dataset) {
      float score = query.dot(datapoint);
      if (score > best) {
        (*answers)[outer_counter] = inner_counter;
        best = score;
      }
      ++inner_counter;
    }
    ++outer_counter;
  }
}

/*
 * Generates answers for the queries using the (optimized) linear scan.
 */
void gen_answers(const vector<Point> &dataset, const vector<Point> &queries,
                 vector<int> *answers) {
  answers->resize(queries.size());
  int outer_counter = 0;
  for (const auto &query : queries) {
    float best = -10.0;
    int inner_counter = 0;
    for (const auto &datapoint : dataset) {
      float score = query.dot(datapoint);
      if (score > best) {
        (*answers)[outer_counter] = inner_counter;
        best = score;
      }
      ++inner_counter;
    }
    ++outer_counter;
  }
}

/*
 * Computes the probability of success using a given number of probes.
 */
double evaluate_num_probes(LSHNearestNeighborTable<Point> *table,
                           const vector<Point> &queries,
                           const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  int outer_counter = 0;
  int num_matches = 0;
  vector<int32_t> candidates;
  for (const auto &query : queries) {
    query_object->get_candidates_with_duplicates(query, &candidates);
    for (auto x : candidates) {
      if (x == answers[outer_counter]) {
        ++num_matches;
        break;
      }
    }
    ++outer_counter;
  }
  return (num_matches + 0.0) / (queries.size() + 0.0);
}

/*
 * Queries the data structure using a given number of probes.
 * It is much slower than 'evaluate_num_probes' and should be used to
 * measure the time.
 */
pair<double, QueryStatistics> evaluate_query_time(
    LSHNearestNeighborTable<Point> *table, const vector<Point> &queries,
    const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  query_object->reset_query_statistics();
  int outer_counter = 0;
  int num_matches = 0;
  int i = 0;
  int topk = 5;
  vector<int> qa;
  for (const auto &query : queries) {
//    auto ans = query_object->find_nearest_neighbor(query);
    query_object->find_k_nearest_neighbors(query, topk, &qa);
    cout << "the " << i ++ << "th query answer is: " <<endl;//<< ans << endl;
    for (auto j = 0; j < qa.size(); ++ j) {
      cout << qa[j] << " ";
    }
    cout << endl;
    if ( qa[0] == answers[outer_counter]) {
      ++num_matches;
    }
    ++outer_counter;
  }
  return make_pair((num_matches + 0.0) / (queries.size() + 0.0),
                   query_object->get_query_statistics());
}

/*
 * Finds the smallest number of probes that gives the probability of success
 * at least 0.9 using binary search.
 */
int find_num_probes(LSHNearestNeighborTable<Point> *table,
                    const vector<Point> &queries, const vector<int> &answers,
                    int start_num_probes) {
  int num_probes = start_num_probes;
  for (;;) {
    cout << "trying " << num_probes << " probes" << endl;
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    if (precision >= 0.9) {
      break;
    }
    num_probes *= 2;
  }

  int r = num_probes;
  int l = r / 2;

  while (r - l > 1) {
    int num_probes = (l + r) / 2;
    cout << "trying " << num_probes << " probes" << endl;
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    if (precision >= 0.9) {
      r = num_probes;
    } else {
      l = num_probes;
    }
  }

  return r;
}

int main() {
  try {
    vector<Point> dataset, queries;
    vector<int> answers;
    int dim, n_total;

    // read the dataset
    float *read_data = (float*) malloc(512000000);
    cout << "reading points" << endl;
//    read_dataset(FILE_NAME, &dataset);
    LoadData(FILE_PATH + FILE_NAME, read_data, "train", dim, n_total);
    cout << "LoadData Done" << endl;
    cout << "dim: " << dim << " n_total: " << n_total << std::endl;
    ReadData(read_data, &dataset, dim, n_total);
    cout << dataset.size() << " points read" << endl;

    // normalize the data points
    cout << "normalizing points" << endl;
    normalize(&dataset);
    cout << "done" << endl;

    // find the center of mass
    Point center = dataset[0];
    for (size_t i = 1; i < dataset.size(); ++i) {
      center += dataset[i];
    }
    center /= dataset.size();

    // selecting NUM_QUERIES data points as queries
    cout << "selecting " << NUM_QUERIES << " queries" << endl;
//    gen_queries(&dataset, &queries);
    GenQueries(&dataset, &queries);
    cout << "done" << endl;

    // running the linear scan
    cout << "running linear scan (to generate nearest neighbors)" << endl;
    auto t1 = high_resolution_clock::now();
    GenAnswers(dataset, queries, &answers);
    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "Brute Force scan done in " << elapsed_time << " secs." << endl;
    cout << elapsed_time / queries.size() << " s per query" << endl;

    // re-centering the data to make it more isotropic
    cout << "re-centering" << endl;
    for (auto &datapoint : dataset) {
      datapoint -= center;
    }
    for (auto &query : queries) {
      query -= center;
    }
    cout << "Re-centering done" << endl;

    // setting parameters and constructing the table
    LSHConstructionParameters params;
    params.dimension = dataset[0].size();
    params.lsh_family = LSHFamily::CrossPolytope;
    params.l = NUM_HASH_TABLES;
    params.distance_function = DistanceFunction::EuclideanSquared;
    cout << "before compute_number_of_hash_function, params.last_cp_dim: " << params.last_cp_dimension << endl;
    compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    params.num_rotations = NUM_ROTATIONS;
    // we want to use all the available threads to set up
    params.num_setup_threads = 0;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    /*
      For an easy way out, you could have used the following.

      LSHConstructionParameters params
        = get_default_parameters<Point>(dataset.size(),
                                   dataset[0].size(),
                                   DistanceFunction::EuclideanSquared,
                                   true);
    */
    cout << "before build index, params.last_cp_dim: " << params.last_cp_dimension << endl;
    cout << "building the index based on the cross-polytope LSH" << endl;
    t1 = high_resolution_clock::now();
    auto table = construct_table<Point>(dataset, params);
    t2 = high_resolution_clock::now();
    elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "Build Index done" << endl;
    cout << "construction time: " << elapsed_time << endl;
    cout << "after build index, params.last_cp_dim: " << params.last_cp_dimension << endl;
//    cout << "after build index, params.num_of_hash_bits: " << params.num_of_hash_bits << endl;

    // finding the number of probes via the binary search
//    cout << "finding the appropriate number of probes" << endl;
//    int num_probes = find_num_probes(&*table, queries, answers, params.l);
//    cout << "done" << endl;
//    cout << num_probes << " probes" << endl;
    int num_probes = 50;

    // executing the queries using the found number of probes to gather
    // statistics
    auto tmp = evaluate_query_time(&*table, queries, answers, num_probes);
    auto score = tmp.first;
    auto statistics = tmp.second;
    cout << "average total query time: " << statistics.average_total_query_time
         << endl;
    cout << "average lsh time: " << statistics.average_lsh_time << endl;
    cout << "average hash table time: " << statistics.average_hash_table_time
         << endl;
    cout << "average distance time: " << statistics.average_distance_time
         << endl;
    cout << "average number of candidates: "
         << statistics.average_num_candidates << endl;
    cout << "average number of unique candidates: "
         << statistics.average_num_unique_candidates << endl;
    cout << "score: " << score << endl;
    free(read_data);
  } catch (runtime_error &e) {
    cerr << "Runtime error: " << e.what() << endl;
    return 1;
  } catch (exception &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "ERROR" << endl;
    return 1;
  }
  return 0;
}
