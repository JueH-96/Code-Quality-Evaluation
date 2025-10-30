# YOUR CODE HERE
H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

min_operations = float('inf')

# Check horizontal sequences
for i in range(H):
    for j in range(W - K + 1):
        # Check sequence from (i, j) to (i, j+K-1)
        operations = 0
        valid = True
        
        for k in range(K):
            cell = grid[i][j + k]
            if cell == 'x':
                valid = False
                break
            elif cell == '.':
                operations += 1
        
        if valid:
            min_operations = min(min_operations, operations)

# Check vertical sequences
for i in range(H - K + 1):
    for j in range(W):
        # Check sequence from (i, j) to (i+K-1, j)
        operations = 0
        valid = True
        
        for k in range(K):
            cell = grid[i + k][j]
            if cell == 'x':
                valid = False
                break
            elif cell == '.':
                operations += 1
        
        if valid:
            min_operations = min(min_operations, operations)

if min_operations == float('inf'):
    print(-1)
else:
    print(min_operations)