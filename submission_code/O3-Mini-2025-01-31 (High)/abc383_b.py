def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = data[3:]
    
    # Collect all floor cells (cells with '.')
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    n = len(floor_cells)
    
    # Precompute coverage for each floor cell when placing a humidifier on that cell.
    # coverage[i] will be a bitmask representing which floor cells are within Manhattan distance D.
    # The j-th bit is set if floor_cells[j] is within distance D from floor_cells[i].
    coverage = [0] * n
    for i, (r, c) in enumerate(floor_cells):
        mask = 0
        for j, (r2, c2) in enumerate(floor_cells):
            if abs(r - r2) + abs(c - c2) <= D:
                mask |= (1 << j)
        coverage[i] = mask

    # Try all pairs of distinct floor cell placements (as the humidifier positions)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            # The combined coverage when humidifiers are placed at floor_cells[i] and floor_cells[j]
            union_mask = coverage[i] | coverage[j]
            # Count the number of humidified floor cells (bits set in union_mask)
            count = bin(union_mask).count("1")
            if count > best:
                best = count

    print(best)

if __name__ == '__main__':
    main()