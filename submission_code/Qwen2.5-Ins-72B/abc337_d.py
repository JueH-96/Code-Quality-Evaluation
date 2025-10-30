def min_operations_to_form_sequence(H, W, K, grid):
    def is_possible_horizontal(i, j, K):
        count = 0
        for k in range(K):
            if grid[i][j + k] == 'o':
                count += 1
        return K - count

    def is_possible_vertical(i, j, K):
        count = 0
        for k in range(K):
            if grid[i + k][j] == 'o':
                count += 1
        return K - count

    min_ops = float('inf')

    for i in range(H):
        for j in range(W - K + 1):
            ops = is_possible_horizontal(i, j, K)
            if ops < min_ops:
                min_ops = ops

    for j in range(W):
        for i in range(H - K + 1):
            ops = is_possible_vertical(i, j, K)
            if ops < min_ops:
                min_ops = ops

    if min_ops == float('inf'):
        return -1
    return min_ops

# Read input
import sys
input = sys.stdin.read
data = input().split()
H, W, K = map(int, data[:3])
grid = data[3:]

# Convert grid to a list of lists
grid = [list(row) for row in grid]

# Solve the problem
result = min_operations_to_form_sequence(H, W, K, grid)

# Print the result
print(result)