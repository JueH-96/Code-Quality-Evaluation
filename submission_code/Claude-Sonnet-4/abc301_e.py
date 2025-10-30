from collections import deque
import sys

def solve():
    H, W, T = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Find start, goal, and candies
    start = None
    goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # BFS to find shortest distance between any two points
    def bfs_distance(start_pos, end_pos):
        if start_pos == end_pos:
            return 0
        
        queue = deque([(start_pos[0], start_pos[1], 0)])
        visited = set()
        visited.add(start_pos)
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c, dist = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] != '#':
                    if (nr, nc) == end_pos:
                        return dist + 1
                    
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        
        return float('inf')
    
    # Check if we can reach goal at all
    min_dist_to_goal = bfs_distance(start, goal)
    if min_dist_to_goal > T:
        print(-1)
        return
    
    num_candies = len(candies)
    if num_candies == 0:
        print(0)
        return
    
    # Precompute distances between all important points
    points = [start, goal] + candies
    n_points = len(points)
    dist = [[float('inf')] * n_points for _ in range(n_points)]
    
    for i in range(n_points):
        for j in range(n_points):
            dist[i][j] = bfs_distance(points[i], points[j])
    
    # DP with bitmask
    # dp[mask] = minimum moves to collect candies in mask and end up at goal
    dp = {}
    
    # Try all possible subsets of candies
    max_candies = 0
    
    for mask in range(1 << num_candies):
        # For each mask, find minimum moves to collect these candies and reach goal
        if mask == 0:
            # No candies collected, just go to goal
            if dist[0][1] <= T:
                max_candies = max(max_candies, 0)
            continue
        
        min_moves = float('inf')
        
        # Try all possible orders to visit the candies in this mask
        candies_in_mask = []
        for i in range(num_candies):
            if mask & (1 << i):
                candies_in_mask.append(i)
        
        # Use DP to find minimum moves for this mask
        # dp_state[pos][submask] = minimum moves to collect submask and be at pos
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def min_moves_for_mask(pos, submask):
            if submask == 0:
                return dist[0][pos] if pos != 0 else 0
            
            result = float('inf')
            for i in range(num_candies):
                if submask & (1 << i):
                    candy_pos = i + 2  # +2 because start=0, goal=1, candies start from 2
                    prev_submask = submask ^ (1 << i)
                    
                    # Try coming from each possible previous position
                    for prev_pos in range(n_points):
                        if prev_pos == 1:  # Can't come from goal
                            continue
                        if prev_submask == 0 and prev_pos != 0:  # If no candies collected, must come from start
                            continue
                        if prev_submask != 0 and prev_pos == 0:  # If candies collected, can't be at start
                            continue
                        
                        prev_moves = min_moves_for_mask(prev_pos, prev_submask)
                        if prev_moves != float('inf'):
                            result = min(result, prev_moves + dist[prev_pos][candy_pos])
            
            return result
        
        # Find minimum moves to collect all candies in mask and be at any candy position
        best_moves = float('inf')
        for i in range(num_candies):
            if mask & (1 << i):
                candy_pos = i + 2
                moves_to_candy = min_moves_for_mask(candy_pos, mask)
                if moves_to_candy != float('inf'):
                    total_moves = moves_to_candy + dist[candy_pos][1]  # +distance to goal
                    best_moves = min(best_moves, total_moves)
        
        if best_moves <= T:
            max_candies = max(max_candies, bin(mask).count('1'))
    
    print(max_candies)

solve()