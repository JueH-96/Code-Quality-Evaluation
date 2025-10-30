def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, D = map(int, data[:3])
    grid_data = data[3:]
    
    # Parse the grid
    grid = [grid_data[i] for i in range(H)]
    
    # Collect all floor cells (using 0-based indexing)
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    # Function to calculate Manhattan distance
    def manhattan_dist(r1, c1, r2, c2):
        return abs(r1 - r2) + abs(c1 - c2)
    
    max_humidified = 0
    
    # Brute force all pairs of floor cells
    n = len(floor_cells)
    for i in range(n):
        for j in range(i+1, n):
            # humidifiers placed on floor_cells[i] and floor_cells[j]
            h1 = floor_cells[i]
            h2 = floor_cells[j]
            
            # Count how many floor cells become humidified
            count = 0
            for fr, fc in floor_cells:
                # Check distance to either of the humidifiers
                d1 = manhattan_dist(fr, fc, h1[0], h1[1])
                d2 = manhattan_dist(fr, fc, h2[0], h2[1])
                if d1 <= D or d2 <= D:
                    count += 1
            
            max_humidified = max(max_humidified, count)
    
    print(max_humidified)

# Do not forget to call main() function
if __name__ == "__main__":
    main()