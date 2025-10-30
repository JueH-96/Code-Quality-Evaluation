# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    visited = set()
    
    def is_valid(r, c):
        return 0 <= r < n and 0 <= c < m and grid[r][c] == '.'
    
    def dfs(r, c):
        visited.add((r, c))
        
        # Move up
        nr, nc = r - 1, c
        while is_valid(nr, nc):
            visited.add((nr, nc))
            nr -= 1
            
        # Move down
        nr, nc = r + 1, c
        while is_valid(nr, nc):
            visited.add((nr, nc))
            nr += 1
        
        # Move left
        nr, nc = r, c - 1
        while is_valid(nr, nc):
            visited.add((nr, nc))
            nc -= 1
            
        # Move right
        nr, nc = r, c + 1
        while is_valid(nr, nc):
            visited.add((nr, nc))
            nc += 1

    dfs(1, 1)
    print(len(visited))

solve()