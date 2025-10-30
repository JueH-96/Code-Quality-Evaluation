# YOUR CODE HERE
def count_reachable_ice(grid):
    n, m = len(grid), len(grid[0])
    visited = [[False] * m for _ in range(n)]
    
    def dfs(i, j):
        if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] == '#' or visited[i][j]:
            return 0
        
        visited[i][j] = True
        count = 1
        
        # Move in all four directions until hitting a rock
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i, j
            while 0 <= ni + di < n and 0 <= nj + dj < m and grid[ni + di][nj + dj] != '#':
                ni += di
                nj += dj
                if not visited[ni][nj]:
                    count += dfs(ni, nj)
        
        return count

    return dfs(1, 1)

# Read input
n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

# Solve and print the result
print(count_reachable_ice(grid))