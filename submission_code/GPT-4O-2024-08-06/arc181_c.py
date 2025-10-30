# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Initialize the grid with '0's
    grid = [['0'] * N for _ in range(N)]
    
    # Fill the grid according to the permutation P
    for i in range(N):
        row = P[i] - 1  # Convert 1-based index to 0-based
        for j in range(i + 1):
            grid[row][j] = '1'
    
    # Fill the grid according to the permutation Q
    for j in range(N):
        col = Q[j] - 1  # Convert 1-based index to 0-based
        for i in range(j + 1):
            grid[i][col] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

main()