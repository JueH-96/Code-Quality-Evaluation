def main():
    import sys, heapq, bisect
    input = sys.stdin.readline

    H, W, Y = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(H)]
    
    # We'll compute the flood threshold for each cell using multi-source Dijkstra.
    INF = 10**9
    flood = [[INF]*W for _ in range(H)]
    heap = []
    
    # Seed the boundary cells: if the cell is on boundary, it is adjacent to sea.
    # Its flood threshold is just its elevation.
    for i in range(H):
        for j in range(W):
            if i == 0 or i == H-1 or j == 0 or j == W-1:
                # cell is adjacent to 'sea' (outside the island)
                if flood[i][j] > grid[i][j]:
                    flood[i][j] = grid[i][j]
                    heapq.heappush(heap, (grid[i][j], i, j))
                    
    # Directions: up, down, left, right.
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    while heap:
        cost, i, j = heapq.heappop(heap)
        # if we have already improved then continue.
        if cost != flood[i][j]:
            continue
        for di, dj in directions:
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W:
                # The water must rise to at least cost, and then the neighbor is flooded if 
                # neighbor's elevation does not exceed the current water level.
                new_cost = max(cost, grid[ni][nj])
                if new_cost < flood[ni][nj]:
                    flood[ni][nj] = new_cost
                    heapq.heappush(heap, (new_cost, ni, nj))
    
    # Now each cell sinks when r (the water level) >= its threshold.
    # We want, for each water level r (year r), the number of cells that remain (threshold > r).
    thresholds = []
    for i in range(H):
        thresholds.extend(flood[i])
    thresholds.sort()
    total = H * W

    out_lines = []
    
    # Process Y queries: for year r (water level = r)
    # (Note: water level = number of years because sea level rises 1 per year.)
    for _ in range(Y):
        r = int(input().strip())
        # find the number of cells with threshold <= r.
        # bisect_right gives us the index of the first element > r.
        count_flooded = bisect.bisect_right(thresholds, r)
        remain = total - count_flooded
        out_lines.append(str(remain))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()