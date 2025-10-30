# YOUR CODE HERE
H, W, Q = map(int, input().split())
grid = [[1] * W for _ in range(H)]
for _ in range(Q):
    R, C = map(int, input().split())
    R -= 1
    C -= 1
    if grid[R][C] == 1:
        grid[R][C] = 0
    else:
        up, down, left, right = -1, -1, -1, -1
        for i in range(R - 1, -1, -1):
            if grid[i][C] == 1:
                up = i
                break
        for i in range(R + 1, H):
            if grid[i][C] == 1:
                down = i
                break
        for j in range(C - 1, -1, -1):
            if grid[R][j] == 1:
                left = j
                break
        for j in range(C + 1, W):
            if grid[R][j] == 1:
                right = j
                break
        if up != -1:
            grid[up][C] = 0
        if down != -1:
            grid[down][C] = 0
        if left != -1:
            grid[R][left] = 0
        if right != -1:
            grid[R][right] = 0
count = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == 1:
            count += 1
print(count)