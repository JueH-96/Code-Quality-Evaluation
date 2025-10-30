def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    H, W, T = map(int, input_data[:3])
    grid_strings = input_data[3:]
    
    # Read the grid
    grid = [grid_strings[r] for r in range(H)]
    
    # Identify S, G, and candy positions
    Spos = None
    Gpos = None
    candies = []
    for r in range(H):
        for c in range(W):
            ch = grid[r][c]
            if ch == 'S':
                Spos = (r, c)
            elif ch == 'G':
                Gpos = (r, c)
            elif ch == 'o':
                candies.append((r, c))
    # Number of candies
    c = len(candies)
    
    # Key positions array: index 0 = S, 1..c = candies, c+1 = G
    key_positions = [Spos] + candies + [Gpos]
    c_plus_2 = c + 2  # total "key" spots
    
    # A large "infinity" for our DP/bfs usage
    INF = 10**9
    
    # 1) BFS from each key position to find shortest distances to every other.
    # dist[i][j] = shortest distance from key_positions[i] to key_positions[j].
    dist = [[INF]*(c_plus_2) for _ in range(c_plus_2)]
    
    # Precompute BFS from each key position
    # We'll do BFS in the grid of size H*W.  We'll map (r,c)->index = r*W + c.
    # BFS function that returns a list of distances to each cell index (r*W + c)
    def bfs_from(sr, sc):
        d2d = [-1]*(H*W)
        start_idx = sr*W + sc
        d2d[start_idx] = 0
        q = deque([start_idx])
        while q:
            cur = q.popleft()
            cr, cc = divmod(cur, W)
            base_dist = d2d[cur]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = cr+dr, cc+dc
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#':  # not a wall
                        nxt_idx = nr*W + nc
                        if d2d[nxt_idx] == -1:
                            d2d[nxt_idx] = base_dist + 1
                            q.append(nxt_idx)
        return d2d
    
    # Fill dist array
    for i in range(c_plus_2):
        r_i, c_i = key_positions[i]
        dist2d = bfs_from(r_i, c_i)
        # For j in [0..c+1], read off distance:
        for j in range(c_plus_2):
            r_j, c_j = key_positions[j]
            d = dist2d[r_j*W + c_j]
            if d >= 0:
                dist[i][j] = d
            else:
                dist[i][j] = INF
    
    # If we cannot go directly S->G at all, it's impossible
    if dist[0][c+1] == INF:
        print(-1)
        return
    
    # We will run a TSP-like DP over the candies to see how many we can visit.
    # But first, check the trivial route: S->G with 0 candies
    answer = -1
    if dist[0][c+1] <= T:
        answer = 0  # we can at least go from S to G with no candies
    
    # If there are no candies, we're done
    if c == 0:
        print(answer)  # either 0, or -1 if even direct S->G was > T
        return
    
    # 2) TSP DP with bitmask over candies. Let dp[mask][v] = minimal distance
    #    starting from S, visiting all candies in mask, and ending at candy v.
    # We'll store dp in a 1D array of length (1<<c)*c to save memory.
    # index in dp: index_dp = mask*c + v
    
    dp_size = (1 << c) * c
    dp_tsp = [INF] * dp_size
    
    # A helper to get dp index
    def idx(m, v):
        return (m << 0) * c + v  # just m*c + v
    
    # Initialize dp for single-candy visits: dp[1<<i][i] = dist[S->candy i]
    for i_candy in range(c):
        d_sc = dist[0][i_candy+1]  # S->that candy
        if d_sc < INF:
            dp_tsp[idx((1 << i_candy), i_candy)] = d_sc
    
    # Precompute popcount for all masks
    popc = [0] * (1 << c)
    for m in range(1, 1 << c):
        popc[m] = popc[m >> 1] + (m & 1)
    
    # Fill DP - O(c^2 * 2^c) transitions
    for mask in range(1 << c):
        base_offset = mask * c
        for v in range(c):
            # check if v is in mask
            if (mask >> v) & 1 == 0:
                continue
            cost_v = dp_tsp[base_offset + v]
            if cost_v == INF or cost_v > T:
                continue
            # Try to go to a new candy w not in mask
            for w in range(c):
                if (mask >> w) & 1 == 0:  # w not in mask
                    d_vw = dist[v+1][w+1]  # candy v -> candy w
                    if d_vw == INF:
                        continue
                    new_mask = mask | (1 << w)
                    new_cost = cost_v + d_vw
                    idx_new = new_mask*c + w
                    if new_cost < dp_tsp[idx_new]:
                        dp_tsp[idx_new] = new_cost
    
    # Now find the maximum number of candies we can visit while finishing at G
    # Check finishing cost for dp[mask][v] + dist[v->G].
    for mask in range(1 << c):
        bitcount = popc[mask]
        base_offset = mask * c
        for v in range(c):
            if (mask >> v) & 1 == 1:
                cost_v = dp_tsp[base_offset + v]
                if cost_v < INF:
                    finish_cost = cost_v + dist[v+1][c+1]
                    if finish_cost <= T:
                        # we can finish at G with bitcount candies
                        if bitcount > answer:
                            answer = bitcount
    
    print(answer if answer != -1 else -1)

# Do not forget to call main()
if __name__ == "__main__":
    main()