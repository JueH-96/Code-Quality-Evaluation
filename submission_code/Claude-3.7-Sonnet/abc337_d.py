def min_operations(H, W, K, grid):
    min_ops = float('inf')
    
    # Check horizontal sequences
    for i in range(H):
        for j in range(W - K + 1):
            operations = 0
            valid = True
            for l in range(K):
                if grid[i][j+l] == 'x':
                    valid = False
                    break
                elif grid[i][j+l] == '.':
                    operations += 1
            if valid:
                min_ops = min(min_ops, operations)
    
    # Check vertical sequences
    for i in range(H - K + 1):
        for j in range(W):
            operations = 0
            valid = True
            for l in range(K):
                if grid[i+l][j] == 'x':
                    valid = False
                    break
                elif grid[i+l][j] == '.':
                    operations += 1
            if valid:
                min_ops = min(min_ops, operations)
    
    if min_ops == float('inf'):
        return -1
    return min_ops

H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]
print(min_operations(H, W, K, grid))