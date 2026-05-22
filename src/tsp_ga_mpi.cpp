#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mpi.h>

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

double fitness(const int* path, const std::vector<City>& cities, double new_city_reward = 1) {
    std::vector<bool> visited(N_CITIES, false);
    double path_length = 0;
    int origin_city = path[0];
    int last_stop = path[0];
    double reward = 0;
    int city_id = 0;
    // go through the path and calculate total distance
    for (int i = 0; i < LEN_PATH; ++i) {
        city_id = path[i];
        // calculate rewards from unique cities visited
        if (!visited[city_id]) {
            reward += new_city_reward;
        }
        visited[city_id] = true;
        path_length += city_distance(cities[city_id], cities[last_stop]);
        last_stop = city_id;
    }
    path_length += city_distance(cities[city_id], cities[origin_city]);
    
    return reward - path_length;
}

void breed(const int* path_a, const int* path_b, int* path_c, std::mt19937& rng, int n_mutations = 2) {
    std::uniform_int_distribution<int> random_gene(0, LEN_PATH - 1);
    std::uniform_int_distribution<int> random_city(0, N_CITIES - 1);

    // Copy from path_a initially
    for(int i = 0; i < LEN_PATH; i++) path_c[i] = path_a[i];
    
    std::vector<int> genes_from_path_b(LEN_PATH / 2);
    for (int i = 0; i < LEN_PATH / 2; i++) {
        genes_from_path_b[i] = random_gene(rng);
    }

    // combine genes
    for (auto gene : genes_from_path_b) {
        path_c[gene] = path_b[gene];
    }

    // choose which genes are mutated
    std::vector<int> mutated_genes(n_mutations);
    for (int i = 0; i < n_mutations; i++) {
        mutated_genes[i] = random_gene(rng);
    }
    
    // mutate
    for (auto mutated_gene : mutated_genes) {
        path_c[mutated_gene] = random_city(rng);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // unique RNG for each process
    std::mt19937 rng(42 + rank);

    // build cities in a circle
    std::vector<City> cities(N_CITIES);
    double angle = 0.0;
    for (int i = 0; i < N_CITIES; i++) {
        angle = (2.0 * 3.141592 * i) / N_CITIES;
        cities[i].x = std::cos(angle);
        cities[i].y = std::sin(angle);
    }

    // split salesmen to each process
    int local_n = N_SALESMEN / size;
    int remainder = N_SALESMEN % size;
    int local_start = rank * local_n + std::min(rank, remainder);
    int local_count = local_n + (rank < remainder ? 1 : 0);

    // gather metadata on root
    std::vector<int> recvcounts_paths(size), displs_paths(size);
    std::vector<int> recvcounts_fitness(size), displs_fitness(size);
    
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int r_count = N_SALESMEN / size + (r < remainder ? 1 : 0);
            int r_start = r * (N_SALESMEN / size) + std::min(r, remainder);
            recvcounts_fitness[r] = r_count;
            displs_fitness[r] = r_start;
            recvcounts_paths[r] = r_count * LEN_PATH;
            displs_paths[r] = r_start * LEN_PATH;
        }
    }

    // using flat data structures
    std::vector<int> paths_flat(N_SALESMEN * LEN_PATH);
    std::vector<int> all_next_paths(N_SALESMEN * LEN_PATH);
    std::vector<double> all_fitness(N_SALESMEN);
    std::vector<int> indices(N_SALESMEN);
    
    std::uniform_int_distribution<int> random_city(0, N_CITIES - 1);
    std::uniform_int_distribution<int> local_parent_dist(0, N_SALESMEN / 2 - 1);

    // generate random sequence of cities
    if (rank == 0) {
        for (int i = 0; i < N_SALESMEN * LEN_PATH; i++) {
            paths_flat[i] = random_city(rng);
        }
    }

    std::vector<int> local_next_paths(local_count * LEN_PATH);
    std::vector<double> local_fitness(local_count);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Main Algorithm
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Broadcast population to all processes
        MPI_Bcast(paths_flat.data(), N_SALESMEN * LEN_PATH, MPI_INT, 0, MPI_COMM_WORLD);

        for (int k = 0; k < local_count; k++) {
            int global_idx = local_start + k;
            
            if (global_idx < 10) {
                std::copy(&paths_flat[global_idx * LEN_PATH], &paths_flat[(global_idx + 1) * LEN_PATH], &local_next_paths[k * LEN_PATH]);
            } else {
                int parent_1 = local_parent_dist(rng);
                int parent_2 = local_parent_dist(rng); 
                breed(&paths_flat[parent_1 * LEN_PATH], &paths_flat[parent_2 * LEN_PATH], &local_next_paths[k * LEN_PATH], rng);
            }
            local_fitness[k] = fitness(&local_next_paths[k * LEN_PATH], cities);
        }

        // Gather paths and fitness scores to root
        MPI_Gatherv(local_next_paths.data(), local_count * LEN_PATH, MPI_INT, all_next_paths.data(), recvcounts_paths.data(), displs_paths.data(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_fitness.data(), local_count, MPI_DOUBLE, all_fitness.data(), recvcounts_fitness.data(), displs_fitness.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // rank 0 sorts fitnesses, writes new paths and prints health checks
        if (rank == 0) {
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&all_fitness](int a, int b) {
                return all_fitness[a] > all_fitness[b];
            });

            for (int j = 0; j < N_SALESMEN; j++) {
                std::copy(&all_next_paths[indices[j] * LEN_PATH], &all_next_paths[(indices[j] + 1) * LEN_PATH], &paths_flat[j * LEN_PATH]);
            }

            if (iter % 1000 == 0) {
                std::cout << "Iteration: " << iter 
                          << "\t | Fitness score: " << fitness(&paths_flat[0], cities) 
                          << "\t| Path: ";
                for (int j = 0; j < LEN_PATH; j++) {
                    std::cout << paths_flat[j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    if (rank == 0) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total Execution Time: " << elapsed.count() << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}