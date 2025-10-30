def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = [list(data[i + 1]) for i in range(N)]
    
    if N == 1:
        # If N is 1, there's nothing to rotate, just print the single element
        print(grid[0][0])
        return
    
    # Collect the outer ring in clockwise order
    outer_ring = []
    
    # Top row (left to right)
    outer_ring.extend(grid[0])
    
    # Right column (top to bottom, excluding the first already included)
    for i in range(1, N):
        outer_ring.append(grid[i][N-1])
    
    # Bottom row (right to left, excluding the last already included)
    for j in range(N-2, -1, -1):
        outer_ring.append(grid[N-1][j])
    
    # Left column (bottom to top, excluding the first and last already included)
    for i in range(N-2, 0, -1):
        outer_ring.append(grid[i][0])
    
    # Rotate the outer ring clockwise by one position
    rotated_outer_ring = [outer_ring[-1]] + outer_ring[:-1]
    
    # Place back the rotated outer ring into a new grid
    new_grid = [row[:] for row in grid]  # make a deep copy of the grid
    
    index = 0
    # Top row
    for j in range(N):
        new_grid[0][j] = rotated_outer_ring[index]
        index += 1
    
    # Right column
    for i in range(1, N):
        new_grid[i][N-1] = rotated_outer_ring[index]
        index += 1
    
    # Bottom row
    for j in range(N-2, -1, -1):
        new_grid[N-1][j] = rotated_outer_ring[index]
        index += 1
    
    # Left column
    for i in range(N-2, 0, -1):
        new_grid[i][0] = rotated_outer_ring[index]
        index += 1
    
    # Print the new grid
    for row in new_grid:
        print(''.join(row))

if __name__ == "__main__":
    main()