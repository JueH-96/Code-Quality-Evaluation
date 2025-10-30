# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))
    
    # Create a grid filled with '0's
    grid = [['0' for _ in range(N)] for _ in range(N)]
    
    # Assign '1's based on the permutations
    for i in range(N):
        for j in range(N):
            if P[i] < P[j]:
                grid[i][j] = '1'
            if Q[j] < Q[i]:
                grid[i][j] = '1'
    
    # Print the grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()