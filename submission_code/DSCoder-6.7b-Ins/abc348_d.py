from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    N = int(input())
    medicines = [tuple(map(int, input().split())) for _ in range(N)]

    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j, 0)
            elif grid[i][j] == 'T':
                goal = (i, j)

    visited = set()
    queue = deque([start])

    while queue:
        i, j, energy = queue.popleft()

        if (i, j) == goal:
            return 'Yes'

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and (ni, nj) not in visited:
                new_energy = energy + 1 if grid[ni][nj] == '.' else energy
                if new_energy > 0:
                    queue.append((ni, nj, new_energy))
                    visited.add((ni, nj))

        for r, c, e in medicines:
            if (i, j) == (r-1, c-1) and energy < e:
                new_energy = energy + e
                if new_energy > 0:
                    queue.append((i, j, new_energy))
                    visited.add((i, j))

    return 'No'

print(solve())