def main():
    import sys
    import heapq

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    Y = int(next(it))
    
    # Read the grid and record the maximum elevation (this will be used later)
    A = [[0] * W for _ in range(H)]
    maxA = 0
    for i in range(H):
        for j in range(W):
            aij = int(next(it))
            A[i][j] = aij
            if aij > maxA:
                maxA = aij

    # We are going to determine, for each cell, the minimum sea level
    # (i.e. the minimum maximum elevation encountered along a path) 
    # from one of the boundaries to that cell. This value f[i][j] is the sea level
    # at which this cell will be flooded.
    #
    # Note that a cell (i,j) floods on year L if and only if sea level L is >= f[i][j].
    # Therefore, for any sea level L, the area of the island that remains is the number
    # of cells for which f[i][j] > L.
    #
    # We use a Dijkstra-like multi-source algorithm on the grid.
    
    INF = 10**9
    f = [[INF] * W for _ in range(H)]
    pq = []
    
    # Initialize the boundary cells (cells adjacent to the sea)
    for i in range(H):
        # Left boundary
        if W > 0:
            if A[i][0] < f[i][0]:
                f[i][0] = A[i][0]
                heapq.heappush(pq, (f[i][0], i, 0))
        # Right boundary
        if W - 1 != 0:
            if A[i][W-1] < f[i][W-1]:
                f[i][W-1] = A[i][W-1]
                heapq.heappush(pq, (f[i][W-1], i, W-1))
                
    for j in range(W):
        # Top boundary
        if H > 0:
            if A[0][j] < f[0][j]:
                f[0][j] = A[0][j]
                heapq.heappush(pq, (f[0][j], 0, j))
        # Bottom boundary
        if H - 1 != 0:
            if A[H-1][j] < f[H-1][j]:
                f[H-1][j] = A[H-1][j]
                heapq.heappush(pq, (f[H-1][j], H-1, j))
    
    # Four possible movement directions (up, down, left, right)
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    # Propagate the flooding thresholds using a Dijkstra-style algorithm.
    # For each neighboring cell, the new flooding threshold is the maximum of
    # the current cell's threshold and the neighbor's own elevation.
    while pq:
        current_th, i, j = heapq.heappop(pq)
        if current_th != f[i][j]:
            continue
        for di, dj in directions:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_th = current_th
                # The neighbor will flood only when the sea level reaches at least its own elevation.
                if A[ni][nj] > new_th:
                    new_th = A[ni][nj]
                # Relax the threshold if we found a way with a lower maximum elevation value.
                if new_th < f[ni][nj]:
                    f[ni][nj] = new_th
                    heapq.heappush(pq, (new_th, ni, nj))
    
    # Now, for each cell we have determined f[i][j], the minimum sea level required
    # for that cell to be flooded (i.e. become part of the sea).
    # Thus, a cell remains above sea level L if and only if f[i][j] > L.
    
    total = H * W
    # Since every cell’s threshold is at most maxA, build a frequency array for thresholds in [0, maxA].
    freq = [0] * (maxA + 1)
    for i in range(H):
        for j in range(W):
            # f[i][j] is between A[i][j] and maxA (inclusive).
            freq[f[i][j]] += 1

    # Build a prefix sum array so that prefix[t] gives the number of cells
    # with flood threshold <= t.
    prefix = [0] * (maxA + 1)
    running = 0
    for t in range(maxA + 1):
        running += freq[t]
        prefix[t] = running
        
    # For each year from 1 to Y:
    # Sea level = year number.
    # Count flooded cells = number of cells with threshold <= sea level.
    # (For sea level > maxA, all cells flood.)
    out_lines = []
    for sea in range(1, Y + 1):
        if sea <= maxA:
            flooded = prefix[sea]
        else:
            flooded = total
        remain = total - flooded
        out_lines.append(str(remain))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()