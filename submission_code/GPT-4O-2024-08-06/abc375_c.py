# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    for i in range(1, N // 2 + 1):
        # Create a temporary grid to store the transformed values
        temp_grid = [row[:] for row in grid]
        
        # Define the range for the current submatrix
        start = i - 1
        end = N - i
        
        # Perform the transformation
        for x in range(start, end + 1):
            for y in range(start, end + 1):
                temp_grid[y][N - 1 - x] = grid[x][y]
        
        # Update the grid with the transformed values
        for x in range(start, end + 1):
            for y in range(start, end + 1):
                grid[x][y] = temp_grid[x][y]
    
    # Print the final grid
    for row in grid:
        print(''.join(row))

main()