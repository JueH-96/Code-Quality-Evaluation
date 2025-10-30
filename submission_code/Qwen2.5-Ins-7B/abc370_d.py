# YOUR CODE HERE
from collections import deque

H, W, Q = map(int, input().split())
grid = [[1] * W for _ in range(H)]
up = [[deque() for _ in range(W)] for _ in range(H)]
down = [[deque() for _ in range(W)] for _ in range(H)]
left = [[deque() for _ in range(W)] for _ in range(H)]
right = [[deque() for _ in range(W)] for _ in range(H)]

def destroy_wall(r, c):
    if grid[r][c] == 1:
        grid[r][c] = 0
        return 1
    else:
        count = 0
        if up[r][c]:
            count += 1
            grid[up[r][c][0]][c] = 0
            up[r][c].popleft()
        if down[r][c]:
            count += 1
            grid[down[r][c][-1]][c] = 0
            down[r][c].pop()
        if left[r][c]:
            count += 1
            grid[r][left[r][c][0]] = 0
            left[r][c].popleft()
        if right[r][c]:
            count += 1
            grid[r][right[r][c][-1]] = 0
            right[r][c].pop()
        return count

for _ in range(Q):
    r, c = map(int, input().split())
    r -= 1
    c -= 1
    count = destroy_wall(r, c)
    if up[r][c]:
        if grid[r][c] == 0:
            up[r][c].appendleft(r)
        else:
            up[r][c].appendleft(r)
    else:
        up[r][c].appendleft(r)
    if down[r][c]:
        if grid[r][c] == 0:
            down[r][c].append(r)
        else:
            down[r][c].append(r)
    else:
        down[r][c].append(r)
    if left[r][c]:
        if grid[r][c] == 0:
            left[r][c].appendleft(c)
        else:
            left[r][c].appendleft(c)
    else:
        left[r][c].appendleft(c)
    if right[r][c]:
        if grid[r][c] == 0:
            right[r][c].append(c)
        else:
            right[r][c].append(c)
    else:
        right[r][c].append(c)

print(sum(sum(row) for row in grid))