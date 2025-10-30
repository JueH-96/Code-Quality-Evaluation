import sys

def solve(N, P, Q):
    grid = [['0'] * N for _ in range(N)]
    
    # Adjust P and Q to be 0-indexed
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Fill the grid
    for i in range(N):
        for j in range(N):
            if P.index(i) < Q.index(j):
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

# Read input
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Solve the problem
solve(N, P, Q)