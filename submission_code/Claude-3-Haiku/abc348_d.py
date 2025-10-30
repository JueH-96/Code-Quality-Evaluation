from collections import deque

def solve():
    # Read input
    H, W = map(int, input().split())
    grid = [list(input()) for _ in range(H)]
    N = int(input())
    medicines = [tuple(map(int, input().split())) for _ in range(N)]

    # Find start and goal positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    # BFS to find if goal is reachable
    queue = deque([(start, 0)])
    visited = set()
    while queue:
        pos, energy = queue.popleft()
        if pos == goal:
            return "Yes"
        if pos in visited:
            continue
        visited.add(pos)
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = pos[0] + di, pos[1] + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#':
                if energy > 0:
                    queue.append(((ni, nj), energy - 1))
                for r, c, e in medicines:
                    if (ni, nj) == (r - 1, c - 1):
                        queue.append(((ni, nj), e))
                        break
    return "No"

print(solve())