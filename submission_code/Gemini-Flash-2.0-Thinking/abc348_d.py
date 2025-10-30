from collections import deque

def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    n = int(input())
    medicines_input = [list(map(int, input().split())) for _ in range(n)]

    medicines = {}
    for r, c, e in medicines_input:
        if (r - 1, c - 1) not in medicines:
            medicines[(r - 1, c - 1)] = []
        medicines[(r - 1, c - 1)].append(e)

    start_row, start_col = -1, -1
    goal_row, goal_col = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_row, start_col = r, c
            elif grid[r][c] == 'T':
                goal_row, goal_col = r, c

    queue = deque([(start_row, start_col, 0)])
    visited = set()

    while queue:
        r, c, energy = queue.popleft()

        if (r, c, energy) in visited:
            continue
        visited.add((r, c, energy))

        if r == goal_row and c == goal_col:
            print("Yes")
            return

        # Move
        if energy > 0:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                    if (nr, nc, energy - 1) not in visited:
                        queue.append((nr, nc, energy - 1))

        # Use medicine
        if (r, c) in medicines:
            for med_energy in medicines[(r, c)]:
                if (r, c, med_energy) not in visited:
                    queue.append((r, c, med_energy))

    print("No")

solve()