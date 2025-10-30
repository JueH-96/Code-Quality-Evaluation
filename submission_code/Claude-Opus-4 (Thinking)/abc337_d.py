# YOUR CODE HERE
H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

min_ops = float('inf')

# Check horizontal sequences
for i in range(H):
    for j in range(W - K + 1):
        # Check if we can create a horizontal sequence starting at (i, j)
        can_create = True
        ops_needed = 0
        
        for k in range(K):
            if grid[i][j + k] == 'x':
                can_create = False
                break
            elif grid[i][j + k] == '.':
                ops_needed += 1
        
        if can_create:
            min_ops = min(min_ops, ops_needed)

# Check vertical sequences
for i in range(H - K + 1):
    for j in range(W):
        # Check if we can create a vertical sequence starting at (i, j)
        can_create = True
        ops_needed = 0
        
        for k in range(K):
            if grid[i + k][j] == 'x':
                can_create = False
                break
            elif grid[i + k][j] == '.':
                ops_needed += 1
        
        if can_create:
            min_ops = min(min_ops, ops_needed)

if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)