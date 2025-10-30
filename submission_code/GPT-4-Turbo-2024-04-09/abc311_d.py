def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = [data[i + 2] for i in range(N)]
    
    # Directions for movement: (row_offset, col_offset)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Set to keep track of all reachable ice squares
    reachable_ice = set()
    
    # Start from (2,2) which is 1-indexed, but we use 0-indexed internally
    start_row, start_col = 1, 1
    reachable_ice.add((start_row, start_col))
    
    # Explore in all four directions
    for dr, dc in directions:
        r, c = start_row, start_col
        while True:
            # Move in the direction until hitting a rock
            nr, nc = r + dr, c + dc
            if grid[nr][nc] == '#':
                break
            # If it's ice, add to reachable and continue
            if grid[nr][nc] == '.':
                reachable_ice.add((nr, nc))
                r, c = nr, nc
    
    # Output the number of reachable ice squares
    print(len(reachable_ice))

if __name__ == "__main__":
    main()