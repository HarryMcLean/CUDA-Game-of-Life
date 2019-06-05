#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>
#include <stdio.h>

using std::cout;
using std::endl;
using std::vector;
using std::ifstream;
using std::swap;
using std::string;
using namespace std;

#define ALIVE 'X'
#define DEAD '-'
#define THREADS 512

__global__
void play_game(int rows, int cols, char d_current_gen[], char d_next_gen[]) {

	int curr_cell = blockIdx.x * blockDim.x + threadIdx.x;

	if (curr_cell < rows * cols) {

		// Row and column indexes of current cell
		int row_idx = curr_cell / cols;
		int col_idx = curr_cell % cols;

		int curr_nbr;
		int nbr_row;
		int nbr_col;

		int num_alive = 0;

		// Loop through to find every neighbour for the current cell
		for (size_t i = row_idx - 1; i <= row_idx + 1; i++) {
			for (size_t j = col_idx - 1; j <= col_idx + 1; j++) {

				nbr_row = i;
				nbr_col = j;

				// Implementing world wrapping
				if (nbr_row < 0) {
					nbr_row += rows;
				}

				if (nbr_col < 0) {
					nbr_col += cols;
				}

				if (nbr_row == rows) {
					nbr_row = 0;
				}

				if (nbr_col == cols) {
					nbr_col = 0;
				}

				// Formula for calculating the current neighbour
				curr_nbr = nbr_row * cols + nbr_col;

				// Continue if the neighbour == the current cell
				if (curr_nbr == curr_cell) {
					continue;
				}

				// Increment count of ALIVE neighbours if the neighbour is ALIVE
				if (d_current_gen[curr_nbr] == ALIVE) {
					num_alive++;
				}
			}
		}

		// If curr_cell is ALIVE
		if (d_current_gen[curr_cell] == ALIVE) {

			// If num live neighbours is < 2 or > 3, kill it
			if (num_alive < 2 || num_alive > 3) {
				d_next_gen[curr_cell] = DEAD;

				// Else if num live neighbours == 2 || == 3
			} else {
				d_next_gen[curr_cell] = ALIVE;
			}

			// Else if curr_cell is DEAD
		} else {

			// If num live neighbours == 3, make it alive
			if (num_alive == 3) {
				d_next_gen[curr_cell] = ALIVE;
			} else {
				d_next_gen[curr_cell] = DEAD;
			}
		}
	}
}

// Function for printing a grid
void print_grid(int rows, int cols, char grid[]) {

	cout << "\n";

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			cout << grid[i * cols + j];
		}
		cout << "\n";
	}
}

int main(int argc, char * argv[]) {

	int opt;
	int num_iterations = 1;
	bool verbose = false;

	string extension = ".txt";

	// Reject the run if no file specified
	if ((string(argv[argc - 1]).find(extension)) == std::string::npos) {
		cout << "Error: You must specify a .txt file as the last parameter (./cugol -i 100 -v input.txt)\n";
		return EXIT_FAILURE;
	}

	while ((opt = getopt(argc, argv, "i:v:")) != -1) {
		switch (opt) {
		case 'i':
			num_iterations = atoi(argv[2]);
			break;
		case 'v':
			verbose = true;
			break;
		}
	}

	int rows = 0;
	int cols = 0;

	// Temporary vector to store unknown amount of characters from input file
	vector<char> temp;

	char c;

	// Read in data to temp
	ifstream fn;

	fn.open(argv[argc - 1], ifstream::in);

	while (!fn.eof()) {

		// If a newline is reached, increment the number of rows
		if (fn.peek() == '\n' || fn.peek() == '\r') {
			rows++;
		}

		fn >> c;
		temp.push_back(c);
	}

	fn.close();

	// We need to increment rows again because it will not find a newline at the end of the file
	rows++;

	cols = temp.size() / rows;

	// Declare array and memory allocation size.
	int array_size = rows * cols;
	const int ARRAY_BYTES = sizeof(char) * array_size;

	// Host arrays and allocation of host memory
	char * h_current_gen = (char *) malloc(ARRAY_BYTES);
	char * h_next_gen = (char *) malloc(ARRAY_BYTES);

	// Copy data from vector to host array.
	for (size_t i = 0; i < array_size; i++) {
		h_current_gen[i] = temp[i];
	}

//	// Printing out the initial state of the game
	print_grid(rows, cols, h_current_gen);

	// Device arrays
	char * d_current_gen;
	char * d_next_gen;

	// Allocate GPU memory
	cudaMalloc((void**) &d_current_gen, ARRAY_BYTES);
	cudaMalloc((void**) &d_next_gen, ARRAY_BYTES);

	// Transfer memory to the GPU
	cudaMemcpy(d_current_gen, h_current_gen, ARRAY_BYTES,
			cudaMemcpyHostToDevice);

	// Loop for as many iterations as was specified (1 if unspecified)
	for (size_t i = 0; i < num_iterations; i++) {

		// Launch kernel
		play_game<<<(array_size + THREADS - 1) / THREADS, THREADS>>>(rows, cols,
				d_current_gen, d_next_gen);

		// Synchronise threads
		cudaDeviceSynchronize();

		// Memcpy back to host and print the new generation if verbose was set
		if (verbose) {
			cudaMemcpy(h_next_gen, d_next_gen, ARRAY_BYTES,
					cudaMemcpyDeviceToHost);

			print_grid(rows, cols, h_next_gen);
		}

		// Pass the memory of d_next_gen over to d_current_gen
		swap(d_current_gen, d_next_gen);
	}

	// Transfer memory from GPU back to host
	cudaMemcpy(h_next_gen, d_next_gen, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// Print final result of game only if verbose was not set
	if (!verbose) {
		print_grid(rows, cols, h_next_gen);
	}

	// Free CPU memory
	free(h_current_gen);
	free(h_next_gen);

	// Free GPU memory
	cudaFree(d_current_gen);
	cudaFree(d_next_gen);

	return EXIT_SUCCESS;
}
