from collections import deque

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    T = int(data[2])
    grid = data[3:H+3]
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find start, goal, and candy positions
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
    
    # Number of candies
    num_candies = len(candies)
    
    # BFS to find shortest path from start to all other points
    def bfs(start):
        dist = [[float('inf')] * W for _ in range(H)]
        queue = deque([start])
        dist[start[0]][start[1]] = 0
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                    if dist[nx][ny] > dist[x][y] + 1:
                        dist[nx][ny] = dist[x][y] + 1
                        queue.append((nx, ny))
        
        return dist
    
    # Distance from start to all points
    dist_from_start = bfs(start)
    
    # If the goal is unreachable within T moves, return -1
    if dist_from_start[goal[0]][goal[1]] > T:
        print(-1)
        return
    
    # Precompute distances between candies and start/goal
    candy_indices = {candies[i]: i for i in range(num_candies)}
    dist_candies = [bfs(candy) for candy in candies]
    
    # DP with BFS to maximize candies collected
    max_candies = -1
    queue = deque([(start[0], start[1], 0, 0)])  # (x, y, candy_mask, moves)
    visited = set()
    visited.add((start[0], start[1], 0))
    
    while queue:
        x, y, candy_mask, moves = queue.popleft()
        
        # Check if we can reach the goal from here within the remaining moves
        if dist_from_start[goal[0]][goal[1]] <= T - moves:
            max_candies = max(max_candies, bin(candy_mask).count('1'))
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                new_moves = moves + 1
                new_candy_mask = candy_mask
                
                # Check if we are on a candy square
                if (nx, ny) in candy_indices:
                    new_candy_mask |= (1 << candy_indices[(nx, ny)])
                
                # If this state is not visited, add to queue
                if (nx, ny, new_candy_mask) not in visited:
                    visited.add((nx, ny, new_candy_mask))
                    queue.append((nx, ny, new_candy_mask, new_moves))
    
    print(max_candies)