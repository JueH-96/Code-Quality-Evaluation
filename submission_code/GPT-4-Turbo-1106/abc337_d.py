def min_operations_to_sequence(H, W, K, grid):
    min_operations = float('inf')
    found = False

    # Check horizontally
    for i in range(H):
        for j in range(W - K + 1):
            operations = 0
            for k in range(K):
                if grid[i][j + k] == 'x':
                    operations = float('inf')
                    break
                elif grid[i][j + k] == '.':
                    operations += 1
            min_operations = min(min_operations, operations)
            if operations != float('inf'):
                found = True

    # Check vertically
    for j in range(W):
        for i in range(H - K + 1):
            operations = 0
            for k in range(K):
                if grid[i + k][j] == 'x':
                    operations = float('inf')
                    break
                elif grid[i + k][j] == '.':
                    operations += 1
            min_operations = min(min_operations, operations)
            if operations != float('inf'):
                found = True

    return -1 if not found else min_operations

# Read input
H, W, K = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Solve the problem
result = min_operations_to_sequence(H, W, K, grid)

# Print the result
print(result)