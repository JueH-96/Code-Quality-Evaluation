from collections import deque

def bfs_max_candies(grid, start, goal, max_moves):
    H, W = len(grid), len(grid[0])
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def bfs(candies_to_ignore):
        queue = deque([(start[0], start[1], 0)])  # row, col, moves
        visited = set()
        while queue:
            r, c, m = queue.popleft()
            if (r, c) == goal:
                return m
            if (r, c, m) in visited or m > max_moves:
                continue
            visited.add((r, c, m))
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and (nr, nc) not in candies_to_ignore:
                    queue.append((nr, nc, m + 1))
        return float('inf')
    
    def find_candies():
        candies = []
        for i in range(H):
            for j in range(W):
                if grid[i][j] == 'o':
                    candies.append((i, j))
        return candies
    
    candies = find_candies()
    max_candies = 0
    for i in range(len(candies) + 1):
        for j in range(i):
            candies_to_ignore = set(candies[:j] + candies[j+1:i])
            moves_needed = bfs(candies_to_ignore)
            if moves_needed <= max_moves:
                max_candies = max(max_candies, i - len(candies_to_ignore))
    
    return max_candies

# Read input
H, W, T = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]

# Find start and goal positions
start = goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)

# Solve the problem
max_candies = bfs_max_candies(grid, start, goal, T)

# Output the result
print(max_candies if max_candies > 0 else -1)