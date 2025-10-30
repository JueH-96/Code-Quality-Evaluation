def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    H, W, T = map(int, input_data[:3])
    grid_chars = input_data[3:]
    
    # Parse the grid
    grid = [list(row) for row in grid_chars]
    
    # Collect positions of special squares
    candies = []
    start = None
    goal = None
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # Label the special nodes: 0 -> Start, 1..k -> Candies, k+1 -> Goal
    # We will build a small adjacency matrix of size (k+2) x (k+2)
    # where k = number of candies.
    nodes = [start] + candies + [goal]
    k = len(candies)
    n = k + 2  # total relevant nodes
    
    # Directions for BFS
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Check bounds and walls
    def valid(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] != '#'
    
    # Compute pairwise distances using BFS for each node
    INF = 10**15
    dist = [[INF]*n for _ in range(n)]
    
    for idx in range(n):
        # BFS from nodes[idx]
        (sr, sc) = nodes[idx]
        queue = deque()
        visited = [[False]*W for _ in range(H)]
        visited[sr][sc] = True
        queue.append((sr, sc, 0))
        
        # We'll fill some temporary distances to all squares
        distmap = [[INF]*W for _ in range(H)]
        distmap[sr][sc] = 0
        
        while queue:
            r, c, d = queue.popleft()
            # If this square is one of our special nodes, record distance
            # We'll find which node it is
            # for speed, we won't do a dictionary look-up each time, let's just check any matches
            # but we do know we only have up to n nodes
            # We'll do a quick check if (r,c) matches any node:
            # We'll do so by an index->(r,c) mapping, which we already have in nodes.
            
            # Instead of searching linearly, let's do it directly:
            # (We know we started BFS from node 'idx'. If we find the other node positions, record dist).
            
            # We'll do a linear check. n <= 20 in the worst case, so it's quick:
            for jdx in range(n):
                if (r, c) == nodes[jdx]:
                    dist[idx][jdx] = d
            
            for dr, dc in directions:
                nr, nc = r+dr, c+dc
                if valid(nr, nc) and not visited[nr][nc]:
                    visited[nr][nc] = True
                    distmap[nr][nc] = d+1
                    queue.append((nr, nc, d+1))
    
    # If we cannot reach goal from start in any manner, dist[0][k+1] would be INF
    # but we still need to consider possibly visiting candies in between,
    # so we cannot conclude from just dist[0][k+1]. We must do TSP with all possible routes.
    
    # However, if for every path from start->(some candy)->goal is also impossible,
    # then the TSP DP will also fail. We'll do the TSP DP approach:
    # dp[mask][v] = minimal distance to reach node v having visited the candies in 'mask'.
    # v in [0..k+1], 0=Start, 1..k = candies, k+1=Goal
    # We want min total distance. Then we'll see which mask can yield dp[mask][goal] <= T.
    # The best bitcount of such mask is the answer.
    
    # If no path is feasible, we print -1.
    
    # First, check if any of the relevant distances are truly unreachable
    # (they'll be INF in dist array). If everything is INF from start->goal via any route,
    # the DP will reveal it as well.
    
    # TSP DP
    # dp[mask][v], mask up to 2^k, v in [0..k+1]
    # number states ~ 2^k * (k+2). For k up to 18, that is ~ 2^18 * 20 => ~5 million states.
    # We'll attempt an efficient bottom-up approach.
    
    # Initialize
    dp = [ [INF]*(n) for _ in range(1<<k) ]
    dp[0][0] = 0  # start at node 0 with no candies visited costs 0 distance
    
    # Precompute the bit for each candy node to speed up
    # If w is in [1..k], this is the bit for that candy
    # If v is 0 or k+1, it doesn't correspond to a candy
    candy_bit = [0]*(n)
    for i_c in range(1, k+1):
        candy_bit[i_c] = 1 << (i_c - 1)
    
    # Standard TSP transitions
    # We won't move FROM the goal node once we've arrived, so skip v = k+1 as source:
    for mask in range(1<<k):
        for v in range(n):
            if dp[mask][v] == INF: 
                continue
            if v == k+1:
                # Already at goal, no need to move further
                continue
            base_cost = dp[mask][v]
            for w in range(n):
                if w == v: 
                    continue
                dcost = dist[v][w]
                if dcost == INF:
                    continue
                new_mask = mask | candy_bit[w]
                new_cost = base_cost + dcost
                # Update dp if better
                if new_cost < dp[new_mask][w]:
                    dp[new_mask][w] = new_cost
    
    # Now figure out the best bitcount for which dp[mask][goal] <= T
    ans = -1
    for mask in range(1<<k):
        cost_to_goal = dp[mask][k+1]
        if cost_to_goal <= T:
            # Count how many candies visited in mask
            # bitcount
            c = bin(mask).count("1")
            if c > ans:
                ans = c
    
    print(ans)
    
# Remember to call main() at the end
if __name__ == "__main__":
    main()