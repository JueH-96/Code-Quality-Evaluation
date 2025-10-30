def check_horizontal(grid, H, W, K):
    min_ops = float('inf')
    for i in range(H):
        for j in range(W-K+1):
            ops = 0
            possible = True
            for k in range(K):
                if grid[i][j+k] == 'x':
                    possible = False
                    break
                elif grid[i][j+k] == '.':
                    ops += 1
            if possible:
                min_ops = min(min_ops, ops)
    return min_ops

def check_vertical(grid, H, W, K):
    min_ops = float('inf')
    for j in range(W):
        for i in range(H-K+1):
            ops = 0
            possible = True
            for k in range(K):
                if grid[i+k][j] == 'x':
                    possible = False
                    break
                elif grid[i+k][j] == '.':
                    ops += 1
            if possible:
                min_ops = min(min_ops, ops)
    return min_ops

H, W, K = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input()))

h_ops = check_horizontal(grid, H, W, K)
v_ops = check_vertical(grid, H, W, K)

min_ops = min(h_ops, v_ops)
if min_ops == float('inf'):
    print(-1)
else:
    print(min_ops)