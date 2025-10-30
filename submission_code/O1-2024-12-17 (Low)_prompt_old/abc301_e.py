from collections import deque
import sys

def solve():
    input_data = sys.stdin.read().strip().split()
    H, W, T = map(int, input_data[:3])
    grid_lines = input_data[3:]
    
    # Read the grid
    grid = [grid_lines[i] for i in range(H)]
    
    # Identify S, G, and candy squares
    start = None
    goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                start = (i, j)
            elif c == 'G':
                goal = (i, j)
            elif c == 'o':
                candies.append((i, j))
    
    # We'll gather "important" squares: S, candy_1..candy_k, G
    # Let node indices be: 0 => S, 1..k => candies, k+1 => G
    # We have at most k <= 18 candies
    special = [start] + candies + [goal]
    
    # BFS function to get distances from one point to all grid cells
    def bfs_from(r, c):
        dist_map = [[-1]*W for _ in range(H)]
        dist_map[r][c] = 0
        queue = deque([(r, c)])
        while queue:
            rr, cc = queue.popleft()
            for nr, nc in [(rr+1, cc),(rr-1,cc),(rr,cc+1),(rr,cc-1)]:
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] != '#' and dist_map[nr][nc] == -1:
                        dist_map[nr][nc] = dist_map[rr][cc] + 1
                        queue.append((nr, nc))
        return dist_map
    
    # Precompute distances between important squares
    k = len(candies)  # number of candies
    n = k + 2         # total special squares (S, candies..., G)
    
    dist = [[-1]*n for _ in range(n)]
    
    # BFS from each important square
    bfs_results = []
    for (r, c) in special:
        bfs_results.append(bfs_from(r, c))
    
    # Fill dist matrix
    for i in range(n):
        for j in range(n):
            (rj, cj) = special[j]
            dist[i][j] = bfs_results[i][rj][cj]
    
    # If we cannot even reach the goal from S directly (or at all), answer might be -1
    # But we'll do the DP approach. If dist is -1, that means no path.
    # We do a TSP with bitmask for candies. dp[mask][i] = minimal distance to reach i having visited 'mask' candies
    # i can be from 0..k (where 0 is S, 1..k are candies), or i = k+1 is G.
    # But we only store dp for i in 0..k (not for G) because from any i we can either move to an unvisited candy or jump to G.
    # Finally we'll check dp[mask][any_i] + dist_to_G.
    
    # Edge case: if dist[0][n-1] == -1, can't go directly from S to G. We still might be able to go via candies if direct path is blocked. We'll keep going.
    # We'll store dp in a 2D array of size [2^k][k+1].  (We won't store for "G"; we'll compute transitions to G on the fly.)
    
    INF = 10**15
    max_mask = 1 << k
    dp = [[INF]*(k+1) for _ in range(max_mask)]
    # dp[0][0] = 0 means we are at S with no candies visited, distance = 0
    dp[0][0] = 0
    
    # We'll compute transitions
    for mask in range(max_mask):
        for i in range(k+1):
            cost_i = dp[mask][i]
            if cost_i == INF:
                continue
            # Attempt to go to the goal
            d_to_goal = dist[i][k+1]  # from node i to G
            if d_to_goal != -1:
                # total distance if we go to G directly
                total_dist = cost_i + d_to_goal
                # number of candies visited = popcount(mask)
                # We'll check at the end all possible (mask, i) -> G transitions
                pass  # We'll handle checking after the loop
            
            # Attempt to go to an unvisited candy j in [1..k]
            for j in range(1, k+1):
                if (mask & (1 << (j-1))) == 0:  # not visited
                    d_ij = dist[i][j]
                    if d_ij == -1:
                        continue
                    new_mask = mask | (1 << (j-1))
                    cand_dist = cost_i + d_ij
                    if cand_dist < dp[new_mask][j]:
                        dp[new_mask][j] = cand_dist
    
    # Now find the best popcount among all dp[mask][i] + dist_to_goal <= T
    answer = -1
    for mask in range(max_mask):
        visited_count = bin(mask).count('1')
        for i in range(k+1):
            cost_i = dp[mask][i]
            if cost_i == INF:
                continue
            d_to_goal = dist[i][k+1]
            if d_to_goal == -1:
                continue
            total_dist = cost_i + d_to_goal
            if total_dist <= T:
                if visited_count > answer:
                    answer = visited_count
    
    print(answer if answer >= 0 else -1)

def main():
    solve()

if __name__ == "__main__":
    main()