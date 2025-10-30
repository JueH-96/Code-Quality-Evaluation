# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_paths(H, W, K, grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    def dfs(x, y, steps):
        if steps == K:
            return 1
        
        count = 0
        grid[x][y] = '#'  # Mark the cell as visited
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                count += dfs(nx, ny, steps + 1)
        grid[x][y] = '.'  # Unmark the cell
        return count
    
    total_paths = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_paths += dfs(i, j, 0)
    
    return total_paths

def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = []
    index = 3
    for i in range(H):
        grid.append(list(data[index + i]))
    
    result = count_paths(H, W, K, grid)
    print(result)

if __name__ == "__main__":
    main()