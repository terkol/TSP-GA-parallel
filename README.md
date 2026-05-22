# Traveling Salesperson Problem via Genetic Algorithm

This repository contains three implementations of a Genetic Algorithm (GA) designed to solve a simplified instance of the Traveling Salesperson Problem (TSP). The TSP is a combinatorial optimization problem seeking the shortest route visiting a predefined set of locations and returning to the origin. A GA is a search heuristic inspired by biological evolution, utilizing operations such as selection, crossover, and mutation to iteratively improve candidate solutions.

## Model Implementations and Applications

The `src/` directory contains three variants of the GA solver. Each relies on a different execution paradigm.

| Model | Paradigm | Application Criteria |
| --- | --- | --- |
| `tsp_ga.cpp` | Single-threaded sequential processing | Applies as a baseline for correctness validation, local debugging, or execution on constrained single-core environments. Lacks parallel scaling capabilities. |
| `tsp_ga_omp.cpp` | Open Multi-Processing (OpenMP) | Applies on a single computational node with multiple CPU cores. OpenMP provides shared-memory parallelization of independent loop iterations (breeding and fitness evaluation) with minimal overhead. |
| `tsp_ga_mpi.cpp` | Message Passing Interface (MPI) | Applies on distributed compute clusters or multi-node systems. MPI provides distributed-memory parallelism by partitioning the population workload across isolated processes. |

## Algorithm Methodology and Underlying Mechanisms

The objective process relies on optimizing a composite fitness score $F$, defined as $F = R - D$. $R$ represents the cumulative reward for visiting unvisited cities, where each unique city yields a base reward of $1.0$, and revisits yield $0$. $D$ is the total Euclidean path length, calculated by summing the distances $d_{a,b} = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}$ between consecutive cities, including the return trip to the origin. The cities are generated programmatically and positioned equidistantly on the circumference of a unit circle using standard trigonometric functions.

The algorithm parameters are statically defined: 100,000 iterations, 30 cities, a population size of 50, and a chromosome length of 30.

Generational advancement utilizes an elitism strategy, where the top 20% of individuals with the highest fitness scores are preserved unmodified. The remaining 80% of the subsequent generation are produced via crossover and mutation. Parent candidate selection is restricted to the top 50% of the current generation. During crossover, a child sequence is instantiated as an exact copy of the first parent. Subsequently, half of the positions are randomly selected and overwritten by the corresponding sequence values from the second parent. Mutation is applied by uniformly selecting two positions within the child chromosome and replacing them with randomly generated city identifiers.

## Performance Results

The following execution times represent objective evidence gathered from running the three implementations under consistent environmental parameters.

| Implementation | Compute Resources | Total Execution Time (ms) |
| --- | --- | --- |
| Sequential | 1 Core | 26196 |
| OpenMP | 12 Cores (Shared Memory) | 7215 |
| MPI | 12 Cores (Distributed Memory) | 6565 |

The empirical data demonstrates a 72.4% reduction in execution time using OpenMP and a 74.9% reduction using MPI with the same amount of cores, relative to the sequential baseline. It is inferred that the fitness evaluation and breeding phases are highly parallelizable, dominating the computational cost of the algorithm.

The MPI implementation marginally outperforms OpenMP in this specific test. The underlying mechanism for MPI involves explicit data broadcast (`MPI_Bcast`) and gathering (`MPI_Gatherv`) per iteration. While this introduces communication overhead that does not exist in the shared-memory OpenMP model, the strict memory isolation and cache locality of MPI processes likely offset the communication penalty at this specific configuration scale. A potential counter-hypothesis is that at significantly larger population sizes or higher iteration frequencies, MPI communication overhead could bottleneck performance, rendering OpenMP faster on a single multi-core node.

## Compilation and Execution

### Sequential (`tsp_ga.cpp`):

```
g++ src/tsp_ga.cpp -o tsp_ga
./tsp_ga
```

### OpenMP (`tsp_ga_omp.cpp`):

```
g++ -fopenmp src/tsp_ga_omp.cpp -o tsp_ga_omp
./tsp_ga_omp
```

### MPI (`tsp_ga_mpi.cpp`):

```
mpic++ src/tsp_ga_mpi.cpp -o tsp_ga_mpi
mpirun -np 12 ./tsp_ga_mpi
```