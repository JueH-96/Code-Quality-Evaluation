import heapq

def solve():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    N = int(input())
    medicines = [list(map(int, input().split())) for _ in range(N)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    energy = [[-1]*W for _ in range(H)]
    energy[start[0]][start[1]] = 0
    for r, c, e in medicines:
        energy[r-1][c-1] = max(energy[r-1][c-1], e)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    queue = [(0, start[0], start[1])]
    while queue:
        e, x, y = heapq.heappop(queue)
        e = -e
        if e < energy[x][y]:
            continue
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy[nx][ny] < e-1:
                energy[nx][ny] = e-1
                heapq.heappush(queue, (-(e-1), nx, ny))
    print('Yes' if energy[goal[0]][goal[1]] >= 0 else 'No')

solve()