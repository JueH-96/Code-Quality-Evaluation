import heapq

def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    n = int(input())
    medicines = []
    for _ in range(n):
        r, c, e = map(int, input().split())
        medicines.append((r - 1, c - 1, e))

    start_r, start_c = -1, -1
    end_r, end_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_r, start_c = r, c
            elif grid[r][c] == 'T':
                end_r, end_c = r, c

    q = [(0, start_r, start_c, set())]  # (energy, row, col, used_medicines)
    visited = set()

    while q:
        energy, r, c, used_medicines = heapq.heappop(q)
        
        if (r,c,tuple(sorted(used_medicines))) in visited:
            continue
        visited.add((r,c,tuple(sorted(used_medicines))))

        if r == end_r and c == end_c:
            print("Yes")
            return

        # Use medicine
        for mr, mc, me in medicines:
            if (mr, mc) == (r, c) and (mr,mc) not in used_medicines:
                new_used_medicines = set(used_medicines)
                new_used_medicines.add((mr,mc))
                heapq.heappush(q, (me, r, c, new_used_medicines))

        # Move
        dr = [-1, 1, 0, 0]
        dc = [0, 0, -1, 1]
        for i in range(4):
            nr, nc = r + dr[i], c + dc[i]
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#' and energy > 0:
                heapq.heappush(q, (energy - 1, nr, nc, used_medicines))

    print("No")

solve()