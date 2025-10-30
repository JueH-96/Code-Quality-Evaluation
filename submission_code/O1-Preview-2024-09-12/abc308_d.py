# YOUR CODE HERE
from collections import deque

H, W = map(int, input().split())
s = [input() for _ in range(H)]
snuke = 'snuke'

visited = [[[False]*5 for _ in range(W)] for _ in range(H)]

queue = deque()

if s[0][0] == 's':
    visited[0][0][0] = True
    queue.append((0, 0, 0))  # (row, col, state)

while queue:
    i, j, state = queue.popleft()
    if i == H -1 and j == W -1:
        print('Yes')
        exit()
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni, nj = i + dx, j + dy
        next_state = (state + 1)%5
        if 0 <= ni < H and 0 <= nj < W:
            if not visited[ni][nj][next_state]:
                if s[ni][nj] == snuke[next_state]:
                    visited[ni][nj][next_state] = True
                    queue.append((ni,nj,next_state))

print('No')