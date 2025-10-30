# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    
    grid = []
    index = 3
    for _ in range(H):
        grid.append(data[index])
        index += 1
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    def dfs(x, y, steps):
        if steps == K:
            nonlocal path_count
            path_count += 1
            return
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and not visited[nx][ny]:
                visited[nx][ny] = True
                dfs(nx, ny, steps + 1)
                visited[nx][ny] = False
    
    path_count = 0
    visited = [[False] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited[i][j] = True
                dfs(i, j, 0)
                visited[i][j] = False
    
    print(path_count)

main()