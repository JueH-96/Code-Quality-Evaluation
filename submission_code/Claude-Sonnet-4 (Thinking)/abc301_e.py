from collections import deque

H, W, T = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find positions
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

# BFS to find shortest distances
def bfs(start_pos):
    dist = {}
    queue = deque([(start_pos, 0)])
    dist[start_pos] = 0
    
    while queue:
        (x, y), d = queue.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = d + 1
                    queue.append(((nx, ny), d + 1))
    
    return dist

# Get distances from start
start_dist = bfs(start)

# Check if goal is reachable
if goal not in start_dist or start_dist[goal] > T:
    print(-1)
else:
    # Get distances from each candy
    candy_dists = []
    for candy in candies:
        candy_dists.append(bfs(candy))
    
    n_candies = len(candies)
    
    if n_candies == 0:
        print(0)
    else:
        # DP with bitmask
        # dp[mask][pos] = minimum moves to visit all candies in mask and end at position pos
        # pos: 0 for start, 1..n_candies for candy index
        
        INF = float('inf')
        dp = [[INF] * (n_candies + 1) for _ in range(1 << n_candies)]
        
        dp[0][0] = 0  # No candies visited, at start
        
        for mask in range(1 << n_candies):
            for pos in range(n_candies + 1):
                if dp[mask][pos] == INF:
                    continue
                
                # Try to visit a new candy
                for i in range(n_candies):
                    if mask & (1 << i):  # Already visited
                        continue
                    
                    new_mask = mask | (1 << i)
                    
                    # Calculate distance from current position to candy i
                    if pos == 0:  # From start
                        if candies[i] in start_dist:
                            new_moves = dp[mask][pos] + start_dist[candies[i]]
                        else:
                            continue
                    else:  # From another candy
                        candy_idx = pos - 1
                        if candies[i] in candy_dists[candy_idx]:
                            new_moves = dp[mask][pos] + candy_dists[candy_idx][candies[i]]
                        else:
                            continue
                    
                    dp[new_mask][i + 1] = min(dp[new_mask][i + 1], new_moves)
        
        # Find the answer
        max_candies = 0
        
        for mask in range(1 << n_candies):
            for pos in range(n_candies + 1):
                if dp[mask][pos] == INF:
                    continue
                
                # Calculate distance to goal
                if pos == 0:  # From start
                    dist_to_goal = start_dist[goal]
                else:  # From candy
                    candy_idx = pos - 1
                    if goal in candy_dists[candy_idx]:
                        dist_to_goal = candy_dists[candy_idx][goal]
                    else:
                        continue
                
                if dp[mask][pos] + dist_to_goal <= T:
                    max_candies = max(max_candies, bin(mask).count('1'))
        
        print(max_candies)