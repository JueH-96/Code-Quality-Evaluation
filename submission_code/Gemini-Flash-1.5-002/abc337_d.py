# YOUR CODE HERE
H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

def count_operations(row, col, direction):
    count = 0
    for i in range(K):
        r, c = row, col
        if direction == 'horizontal':
            c += i
        else:
            r += i
        if 0 <= r < H and 0 <= c < W:
            if grid[r][c] == '.':
                count += 1
        else:
            return float('inf')
    return count

min_operations = float('inf')
for i in range(H):
    for j in range(W):
        min_operations = min(min_operations, count_operations(i, j, 'horizontal'))
        min_operations = min(min_operations, count_operations(i, j, 'vertical'))

if min_operations == float('inf'):
    print(-1)
else:
    print(min_operations)