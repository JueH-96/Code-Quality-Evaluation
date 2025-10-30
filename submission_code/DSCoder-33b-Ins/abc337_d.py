# YOUR CODE HERE
H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

def check_horizontal(grid, H, W, K):
    for i in range(H):
        for j in range(W-K+1):
            if all(grid[i][j+k] == 'o' for k in range(K)):
                return True
    return False

def check_vertical(grid, H, W, K):
    for i in range(H-K+1):
        for j in range(W):
            if all(grid[i+k][j] == 'o' for k in range(K)):
                return True
    return False

if check_horizontal(grid, H, W, K) or check_vertical(grid, H, W, K):
    print(0)
else:
    print(-1)