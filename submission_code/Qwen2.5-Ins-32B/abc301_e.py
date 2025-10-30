import sys
from collections import deque

def solve():
    H, W, T = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Find start, goal, and candy positions
    start = goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # BFS to find shortest path from start to goal
    def bfs(start, goal, T):
        queue = deque([(start, 0, 0)])
        visited = set([start])
        while queue:
            (x, y), dist, candy_count = queue.popleft()
            if (x, y) == goal:
                return dist, candy_count
            if dist > T:
                continue
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_candy_count = candy_count + (1 if grid[nx][ny] == 'o' else 0)
                    queue.append(((nx, ny), dist + 1, new_candy_count))
        return float('inf'), -1
    
    # Calculate shortest path and maximum candy count
    dist, candy_count = bfs(start, goal, T)
    
    if dist > T:
        print(-1)
    else:
        print(candy_count)

solve()