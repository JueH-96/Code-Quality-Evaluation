def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, D = map(int, input_data[:3])
    grid_data = input_data[3:]
    
    grid = grid_data  # Each element in grid is a string representing a row
    
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_humidified = 0
    # Try all pairs of distinct floor cells to place the two humidifiers
    for i1, j1 in floor_cells:
        for i2, j2 in floor_cells:
            if (i1, j1) == (i2, j2):
                continue
            count = 0
            # Count how many floor cells are humidified if humidifiers are at (i1, j1) and (i2, j2)
            for ix, jx in floor_cells:
                dist1 = abs(ix - i1) + abs(jx - j1)
                dist2 = abs(ix - i2) + abs(jx - j2)
                if dist1 <= D or dist2 <= D:
                    count += 1
            max_humidified = max(max_humidified, count)
    
    print(max_humidified)

if __name__ == "__main__":
    main()