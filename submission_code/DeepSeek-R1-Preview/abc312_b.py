n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

# Precompute the coordinates for the regions and adjacent cells
top_left_square = [(a, b) for a in range(3) for b in range(3)]
bottom_right_square = [(a, b) for a in range(6, 9) for b in range(6, 9)]
top_adjacent = [(3, 0), (3, 1), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
bottom_adjacent = [(5, 6), (5, 7), (5, 8), (6, 5), (7, 5), (8, 5), (5, 5)]

results = []

for i in range(1, n - 8 + 1):
    for j in range(1, m - 8 + 1):
        i_grid = i - 1
        j_grid = j - 1
        valid = True
        
        # Check top-left 3x3
        for a, b in top_left_square:
            if grid[i_grid + a][j_grid + b] != '#':
                valid = False
                break
        if not valid:
            continue
        
        # Check bottom-right 3x3
        for a, b in bottom_right_square:
            if grid[i_grid + a][j_grid + b] != '#':
                valid = False
                break
        if not valid:
            continue
        
        # Check top adjacent cells
        for a, b in top_adjacent:
            if grid[i_grid + a][j_grid + b] != '.':
                valid = False
                break
        if not valid:
            continue
        
        # Check bottom adjacent cells
        for a, b in bottom_adjacent:
            if grid[i_grid + a][j_grid + b] != '.':
                valid = False
                break
        if valid:
            results.append((i, j))

# Sort the results lexicographically
results.sort()

for i, j in results:
    print(i, j)