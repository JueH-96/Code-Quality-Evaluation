from collections import deque

def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    n = int(input())
    medicines = []
    for _ in range(n):
        r, c, e = map(int, input().split())
        medicines.append((r - 1, c - 1, e))

    start_r, start_c = -1, -1
    goal_r, goal_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_r, start_c = r, c
            elif grid[r][c] == 'T':
                goal_r, goal_c = r, c

    q = deque([(start_r, start_c, 0, set())])
    visited = set()

    while q:
        r, c, energy, used_medicines = q.popleft()

        if (r, c) == (goal_r, goal_c):
            print("Yes")
            return

        if (r, c, energy, tuple(sorted(used_medicines))) in visited:
            continue
        visited.add((r, c, energy, tuple(sorted(used_medicines))))

        # Try using medicine
        for i, (med_r, med_c, med_e) in enumerate(medicines):
            if (r, c) == (med_r, med_c) and i not in used_medicines:
                q.append((r, c, med_e, used_medicines | {i}))

        # Try moving
        if energy > 0:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                    q.append((nr, nc, energy - 1, used_medicines))
    
    print("No")

solve()