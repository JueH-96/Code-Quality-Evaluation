import sys
from collections import deque

def solve():
    """
    Main function to solve the problem.
    This program finds the maximum number of candies that can be visited on a path
    from a start 'S' to a goal 'G' in a grid, within a given time limit 'T'.

    The solution approach is as follows:
    1.  Model the problem as finding a path in a graph. The nodes of this graph are
        the "special" squares: the start, goal, and all candy squares.
    2.  Calculate the shortest travel time (number of moves) between every pair of
        these special squares. This is done by running a Breadth-First Search (BFS)
        from each special point.
    3.  With the all-pairs shortest paths matrix, the problem reduces to a variant of
        the Traveling Salesperson Problem (TSP). We use dynamic programming with
        bitmasking to solve it, since the number of candies is small.
    4.  The state `dp[mask][i]` stores the minimum time to visit the set of candies
        represented by `mask`, ending at candy `i`.
    5.  A preliminary check ensures that the direct path from S to G is possible within
        the time limit.
    6.  The DP table is filled, and then used to find the path that visits the
        maximum number of candies while respecting the time constraint T.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        H, W, T = map(int, input().split())
        grid = [input().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    # 1. Locate S, G, and all candies 'o'
    s_pos = None
    g_pos = None
    candy_pos = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                s_pos = (r, c)
            elif grid[r][c] == 'G':
                g_pos = (r, c)
            elif grid[r][c] == 'o':
                candy_pos.append((r, c))

    K = len(candy_pos)
    
    # Create a list of all "special" points: S, then candies, then G
    special_points = [s_pos] + candy_pos + [g_pos]
    num_special = len(special_points)
    
    # 2. Compute all-pairs shortest paths between special points using BFS
    
    def bfs(start_r, start_c):
        q = deque([(start_r, start_c, 0)])
        dist_grid = [[-1] * W for _ in range(H)]
        dist_grid[start_r][start_c] = 0
        
        while q:
            r, c, d = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and dist_grid[nr][nc] == -1:
                    dist_grid[nr][nc] = d + 1
                    q.append((nr, nc, d + 1))
        return dist_grid

    dist_matrix = [[-1] * num_special for _ in range(num_special)]
    for i in range(num_special):
        start_r, start_c = special_points[i]
        distances_from_i = bfs(start_r, start_c)
        for j in range(num_special):
            end_r, end_c = special_points[j]
            dist_matrix[i][j] = distances_from_i[end_r][end_c]
            
    inf = float('inf')
    for i in range(num_special):
        for j in range(num_special):
            if dist_matrix[i][j] == -1:
                dist_matrix[i][j] = inf

    # 3. Check if reaching Goal from Start is possible at all within time T
    # Indices: S is 0, Candies are 1 to K, G is K+1
    s_to_g_dist = dist_matrix[0][K + 1]
    if s_to_g_dist > T:
        print(-1)
        return
        
    # 4. Use Dynamic Programming with Bitmasking
    # dp[mask][i]: min cost to visit candies in `mask`, ending at candy `i`.
    dp = [[inf] * K for _ in range(1 << K)]

    # Initialization: Paths from Start to each candy
    for i in range(K):
        cost = dist_matrix[0][i + 1]  # S (idx 0) to candy i (idx i+1)
        if cost <= T:
            dp[1 << i][i] = cost

    # DP transitions
    for mask in range(1, 1 << K):
        for i in range(K):
            if dp[mask][i] > T: # Pruning: path is already too long
                continue
            
            for j in range(K):
                if not ((mask >> j) & 1): # If candy `j` is not yet visited
                    cost_i_to_j = dist_matrix[i + 1][j + 1] # candy i to candy j
                    
                    if cost_i_to_j != inf:
                        new_mask = mask | (1 << j)
                        new_cost = dp[mask][i] + cost_i_to_j
                        
                        if new_cost <= T: # Pruning: only consider paths within time limit
                            dp[new_mask][j] = min(dp[new_mask][j], new_cost)

    # 5. Find the maximum number of candies visited
    ans = 0
    bit_counts = [bin(i).count('1') for i in range(1 << K)]
    
    for mask in range(1 << K):
        num_candies = bit_counts[mask]
        for i in range(K):
            if (mask >> i) & 1: # if candy i is in the current set
                cost_so_far = dp[mask][i]
                if cost_so_far == inf:
                    continue
                
                cost_to_g = dist_matrix[i + 1][K + 1] # candy i to G
                
                if cost_to_g != inf:
                    total_cost = cost_so_far + cost_to_g
                    if total_cost <= T:
                        ans = max(ans, num_candies)
    
    print(ans)

if __name__ == "__main__":
    solve()