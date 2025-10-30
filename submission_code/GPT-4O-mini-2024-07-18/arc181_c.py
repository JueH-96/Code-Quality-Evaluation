def fill_grid(N, P, Q):
    # Initialize an N x N grid with all zeros
    grid = [['0'] * N for _ in range(N)]
    
    # Fill the grid according to the permutations P and Q
    for i in range(N):
        for j in range(N):
            # We will fill the grid in a way that respects the order defined by P and Q
            if (i + j) % 2 == 0:
                grid[P[i] - 1][j] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
P = list(map(int, data[1].split()))
Q = list(map(int, data[2].split()))

fill_grid(N, P, Q)