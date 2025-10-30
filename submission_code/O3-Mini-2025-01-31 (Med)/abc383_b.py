def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = []
    index = 3
    for _ in range(H):
        grid.append(data[index])
        index += 1

    # Gather all floor cell positions.
    floors = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floors.append((i, j))
    n = len(floors)
    
    # Precompute the set of floor cells (using bitmasks) 
    # that become humidified when a humidifier is placed on a given floor cell.
    coverage = []
    for pos in floors:
        r, c = pos
        # Use an integer as bitmask, where the k-th bit set indicates the k-th floor cell is humidified.
        mask = 0
        for k, (r2, c2) in enumerate(floors):
            if abs(r - r2) + abs(c - c2) <= D:
                mask |= (1 << k)
        coverage.append(mask)
        
    max_humidified = 0
    # Try every distinct pair of floor cells for placement of humidifiers.
    for i in range(n):
        for j in range(i+1, n):
            # The union of humidified floors from the two placements.
            union_mask = coverage[i] | coverage[j]
            # Count the number of humidified floor cells using bit_count().
            count_humidified = union_mask.bit_count()
            if count_humidified > max_humidified:
                max_humidified = count_humidified

    sys.stdout.write(str(max_humidified))

if __name__ == '__main__':
    main()