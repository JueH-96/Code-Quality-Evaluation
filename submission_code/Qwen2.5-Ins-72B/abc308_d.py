# YOUR CODE HERE
def can_form_snuke_path(H, W, grid):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    target = "snuke"
    visited = [[False] * W for _ in range(H)]
    
    def dfs(x, y, index):
        if index == len(target):
            return True
        if x < 0 or x >= H or y < 0 or y >= W or visited[x][y] or grid[x][y] != target[index % 5]:
            return False
        visited[x][y] = True
        for dx, dy in directions:
            if dfs(x + dx, y + dy, index + 1):
                return True
        visited[x][y] = False
        return False
    
    return dfs(0, 0, 0)

import sys
input = sys.stdin.read
data = input().split()
H, W = map(int, data[0].split())
grid = data[1:]

if can_form_snuke_path(H, W, grid):
    print("Yes")
else:
    print("No")