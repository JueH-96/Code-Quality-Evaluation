# YOUR CODE HERE
def min_operations(H, W, K, grid):
    def check_horizontal(i, j):
        return all(grid[i][j+k] in 'o.' for k in range(K))
    
    def check_vertical(i, j):
        return all(grid[i+k][j] in 'o.' for k in range(K))
    
    def count_dots(i, j, horizontal):
        if horizontal:
            return sum(1 for k in range(K) if grid[i][j+k] == '.')
        else:
            return sum(1 for k in range(K) if grid[i+k][j] == '.')
    
    min_ops = float('inf')
    
    for i in range(H):
        for j in range(W-K+1):
            if check_horizontal(i, j):
                min_ops = min(min_ops, count_dots(i, j, True))
    
    for i in range(H-K+1):
        for j in range(W):
            if check_vertical(i, j):
                min_ops = min(min_ops, count_dots(i, j, False))
    
    return min_ops if min_ops != float('inf') else -1

H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

result = min_operations(H, W, K, grid)
print(result)