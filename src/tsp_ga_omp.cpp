#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>

#define ITERATIONS 100000
#define N_CITIES 30
#define N_SALESMEN 50
#define LEN_PATH 30
    
struct City { double x, y; };

double city_distance(City city_a, City city_b) {
    double dx = city_a.x - city_b.x;
    double dy = city_a.y - city_b.y;
    return std::sqrt(dx*dx + dy*dy);
}

double fitness(const std::vector<int>& path, const std::vector<City>& cities, double new_city_reward = 1) {
    std::vector<bool> visited(N_CITIES, false);
    double path_length = 0;
    int origin_city = path[0];
    int last_stop = path[0];
    double reward = 0;
    int city_id = 0;
    for (int i = 0; i < LEN_PATH; ++i) {
        city_id = path[i];
        // calculate rewards from visiting cities
        if (visited[city_id] == false) 
            reward += new_city_reward;

        visited[city_id] = true;
        path_length += city_distance(cities[city_id], cities[last_stop]);
        last_stop = city_id;
    }
    path_length += city_distance(cities[city_id], cities[origin_city]); //return home
    
    return  reward - path_length;
}

std::vector<int> breed(std::vector<int>& path_a, std::vector<int>& path_b, std::mt19937& rng, int n_mutations = 2) {
    
    std::uniform_int_distribution<int> random_gene(0,LEN_PATH-1);
    std::uniform_int_distribution<int> random_city(0,N_CITIES-1);

    std::vector<int> path_c = path_a;
    std::vector<int> genes_from_path_b(LEN_PATH/2);
    // choose genes copied from path b
    for (int i = 0; i < LEN_PATH/2; i++)
        genes_from_path_b[i] = random_gene(rng);

    // combine genes
    for (auto gene : genes_from_path_b) 
        path_c[gene] = path_b[gene];

    std::vector<int> mutated_genes(n_mutations);

    // generate which genes are mutated
    for (int i = 0; i<n_mutations; i++)
        mutated_genes[i] = random_gene(rng);
    
    // generate mutations
    for (auto mutated_gene : mutated_genes)
        path_c[mutated_gene] = random_city(rng);

    return path_c;
}

int main() {
    // each thread gets its own rng
    int n_threads = omp_get_max_threads();
    std::mt19937 rng(42);

    // Give each thread its own RNG
    std::vector<std::mt19937> thread_rngs(n_threads);
    for (int i=0;i<n_threads;i++) {
        thread_rngs[i].seed(42+i);
    }

    std::vector<City> cities(N_CITIES);
    double angle = 0.0;

    // Put cities in a circle
    for (int i = 0; i < N_CITIES; i++) {
        angle = (2.0 * 3.141592 * i) / N_CITIES;
        cities[i].x = std::cos(angle);
        cities[i].y = std::sin(angle);
    }

    std::vector<std::vector<int>> paths(N_SALESMEN, std::vector<int>(LEN_PATH));
    std::uniform_int_distribution<int> random_city(0,N_CITIES-1);

    // Give every salesman a random sequence of cities they visit
    for (int i = 0; i<N_SALESMEN; i++) {
        for (int j = 0; j<LEN_PATH; j++) {
             paths[i][j] = random_city(rng);
        }
    }

    std::vector<std::vector<int>> next_paths = paths;
    std::uniform_int_distribution<int> local_parent_dist(0, N_SALESMEN / 2 - 1);
    std::vector<double> fitness_scores(N_SALESMEN);
    std::vector<int> indices(N_SALESMEN);
    
    // Main Algorithm
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {

        // keep the elite (highest fitness)
        for (int j = 0; j < 10; j++) 
            next_paths[j] = paths[j];
        
        // generate new generation
        #pragma omp parallel for
        for (int j = 10; j < N_SALESMEN; j++) {
            int thread_id = omp_get_thread_num();
            int parent_1 = local_parent_dist(thread_rngs[thread_id]);
            int parent_2 = local_parent_dist(thread_rngs[thread_id]); 

            next_paths[j] = breed(paths[parent_1], paths[parent_2], thread_rngs[thread_id]); 
        }

        // Calculate fitness scores
        #pragma omp parallel for
        for (int j = 0; j < N_SALESMEN; j++) 
            fitness_scores[j] = fitness(next_paths[j], cities);
        

        // Sort salesmen by fitness
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&fitness_scores](int a, int b) {
            return fitness_scores[a] > fitness_scores[b];
        });

        for (int j = 0; j < N_SALESMEN; j++) 
            paths[j] = next_paths[indices[j]];
        

        // Periodic health checks
        if (i%1000 == 0) {
            std::cout << "Iteration: " << i << "\t | Fitness score: " << fitness(paths[0], cities) << "\t| Path: ";
            for (int j = 0; j<LEN_PATH; j++)
                std::cout << paths[0][j] << " ";
                std::cout << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    
    std::cout << "Total Execution Time: " << elapsed.count() << " ms" << std::endl;

    return 0;
}