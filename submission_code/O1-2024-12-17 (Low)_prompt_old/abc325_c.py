def solve():
    import sys
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]
    
    # Directions for horizontal, vertical, and diagonal adjacency
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    visited = [[False]*W for _ in range(H)]
    
    def dfs(r, c):
        stack = [(r, c)]
        visited[r][c] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < H and 0 <= ny < W:
                    if not visited[nx][ny] and grid[nx][ny] == '#':
                        visited[nx][ny] = True
                        stack.append((nx, ny))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                count += 1
                dfs(i, j)
                
    print(count)

def main():
    solve()

# Run the solve() function
if __name__ == "__main__":
    main()