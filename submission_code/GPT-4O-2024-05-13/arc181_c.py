# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Create an empty grid
    grid = [['0'] * N for _ in range(N)]
    
    # Fill the grid to satisfy the conditions
    for i in range(N):
        for j in range(N):
            if P[i] <= Q[j]:
                grid[i][j] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

# Call the solve function
solve()