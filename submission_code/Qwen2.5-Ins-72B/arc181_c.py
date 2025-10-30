# YOUR CODE HERE
def solve(N, P, Q):
    grid = [['0'] * N for _ in range(N)]
    
    # Initialize the grid with 0s and 1s based on the permutations
    for i in range(N):
        for j in range(N):
            if P[i] < P[j]:
                grid[i][j] = '1'
            if Q[j] < Q[i]:
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