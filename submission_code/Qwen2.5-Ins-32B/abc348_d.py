import sys
from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    N = int(input())
    medicines = [tuple(map(int, input().split())) for _ in range(N)]
    medicines = [(r-1, c-1, e) for r, c, e in medicines]

    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)

    def bfs(start, medicines):
        visited = set()
        queue = deque([(start, 0, 0)])  # (position, energy, used_medicines)
        while queue:
            (r, c), energy, used_medicines = queue.popleft()
            if (r, c) == goal:
                return True
            if (r, c, used_medicines) in visited:
                continue
            visited.add((r, c, used_medicines))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and energy > 0:
                    queue.append(((nr, nc), energy - 1, used_medicines))
            if (r, c) in medicines and medicines.index((r, c)) not in used_medicines:
                new_used_medicines = used_medicines | (1 << medicines.index((r, c)))
                queue.append(((r, c), medicines[medicines.index((r, c))][2], new_used_medicines))
        return False

    if bfs(start, medicines):
        print("Yes")
    else:
        print("No")

solve()