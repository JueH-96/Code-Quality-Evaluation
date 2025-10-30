import collections
import sys

def solve():
    H, W, T = map(int, sys.stdin.readline().split())
    grid_chars = [sys.stdin.readline().strip() for _ in range(H)]

    S_pos = None
    G_pos = None
    candy_pos_list = []

    for r in range(H):
        for c in range(W):
            if grid_chars[r][c] == 'S':
                S_pos = (r, c)
            elif grid_chars[r][c] == 'G':
                G_pos = (r, c)
            elif grid_chars[r][c] == 'o':
                candy_pos_list.append((r, c))

    K = len(candy_pos_list)
    
    BFS_INF = float('inf') 

    memo_bfs = {} # Memoization for BFS results

    def bfs(start_coord_tuple):
        start_coord = tuple(start_coord_tuple) # Ensure hashable for dict key

        if start_coord in memo_bfs:
            return memo_bfs[start_coord]

        q = collections.deque()
        dist_grid = [[BFS_INF] * W for _ in range(H)]
        
        # Start BFS if the start_coord is valid (not a wall, within bounds)
        # This is guaranteed by problem statement for S, G, 'o'
        if not (0 <= start_coord[0] < H and 0 <= start_coord[1] < W and \
                grid_chars[start_coord[0]][start_coord[1]] != '#'):
            # This case should not be reached given problem constraints
            memo_bfs[start_coord] = dist_grid 
            return dist_grid

        dist_grid[start_coord[0]][start_coord[1]] = 0
        q.append((start_coord, 0))
        
        dr = [-1, 1, 0, 0] # Changes in row for up, down, left, right
        dc = [0, 0, -1, 1] # Changes in col for up, down, left, right

        while q:
            (r, c), d = q.popleft()

            for i_move in range(4): # Iterate over 4 possible moves
                nr, nc = r + dr[i_move], c + dc[i_move]
                if 0 <= nr < H and 0 <= nc < W and \
                   grid_chars[nr][nc] != '#' and \
                   dist_grid[nr][nc] == BFS_INF: # If traversable and not yet visited with shorter path
                    dist_grid[nr][nc] = d + 1
                    q.append(((nr, nc), d + 1))
        
        memo_bfs[start_coord] = dist_grid
        return dist_grid

    # Handle case with no candies
    if K == 0:
        d_from_S_grid = bfs(S_pos)
        cost_S_G = d_from_S_grid[G_pos[0]][G_pos[1]]
        if cost_S_G <= T:
            print(0)
        else:
            print(-1)
        return

    # Precompute distances between key points
    sd = [BFS_INF] * K  # Distances from S to C_k (candy k)
    cg = [BFS_INF] * K  # Distances from C_k to G
    cd = [[BFS_INF] * K for _ in range(K)] # Distances from C_k1 to C_k2
    
    d_from_S_grid = bfs(S_pos)
    sg = d_from_S_grid[G_pos[0]][G_pos[1]] # Distance S to G
    for k_idx in range(K):
        sd[k_idx] = d_from_S_grid[candy_pos_list[k_idx][0]][candy_pos_list[k_idx][1]]

    for k1_idx in range(K):
        d_from_Ck1_grid = bfs(candy_pos_list[k1_idx])
        cg[k1_idx] = d_from_Ck1_grid[G_pos[0]][G_pos[1]]
        for k2_idx in range(K):
            cd[k1_idx][k2_idx] = d_from_Ck1_grid[candy_pos_list[k2_idx][0]][candy_pos_list[k2_idx][1]]
    
    DP_INF = float('inf') 
    # dp[mask][i] = min cost to visit candies in 'mask', ending at candy C_i
    dp = [[DP_INF] * K for _ in range(1 << K)]

    # Initialize DP: paths S -> C_i
    for i_candy in range(K): 
        if sd[i_candy] != BFS_INF: # Check reachability from S to C_i
            dp[1 << i_candy][i_candy] = sd[i_candy]

    # Fill DP table
    for mask in range(1, 1 << K): # Iterate over all possible subsets of candies
        for i_candy in range(K): # C_i is the candy ending the path for 'mask'
            if not ((mask >> i_candy) & 1): # C_i must be in 'mask'
                continue
            
            prev_mask = mask ^ (1 << i_candy) # Mask representing candies visited before C_i
            if prev_mask == 0:
                # Base case: mask is (1 << i_candy), path S -> C_i. This is already initialized.
                continue 

            # Try to extend paths S -> ... -> C_j to S -> ... -> C_j -> C_i
            for j_candy in range(K): # C_j was the candy ending the path for 'prev_mask'
                if not ((prev_mask >> j_candy) & 1): # C_j must be in 'prev_mask'
                    continue
                
                # If path to C_j existed and C_j can reach C_i
                if dp[prev_mask][j_candy] != DP_INF and cd[j_candy][i_candy] != BFS_INF:
                    cost = dp[prev_mask][j_candy] + cd[j_candy][i_candy]
                    if cost < dp[mask][i_candy]: # If this new path is shorter
                        dp[mask][i_candy] = cost
    
    max_candies_visited = -1

    # Check path S -> G (0 candies)
    if sg != BFS_INF and sg <= T:
        max_candies_visited = 0

    # Check paths involving candies: S -> ... (candies in mask) ... -> G
    for mask in range(1, 1 << K):
        num_candies_in_mask = bin(mask).count('1')
        for i_candy in range(K): # Last candy C_i before going to G
            if (mask >> i_candy) & 1: # If C_i is in mask
                # Path: S -> ... (candies in mask, ending C_i) -> G
                # Check if path to C_i exists and C_i can reach G
                if dp[mask][i_candy] != DP_INF and cg[i_candy] != BFS_INF:
                    total_dist = dp[mask][i_candy] + cg[i_candy]
                    if total_dist <= T: # Check against total move limit
                        max_candies_visited = max(max_candies_visited, num_candies_in_mask)
                        
    print(max_candies_visited)

solve()