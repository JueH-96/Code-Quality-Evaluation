# YOUR CODE HERE
import heapq

def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    N = int(input())
    medicines = []
    for _ in range(N):
        r, c, e = map(int, input().split())
        medicines.append((r - 1, c - 1, e))

    start = None
    goal = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'T':
                goal = (r, c)

    q = [(0, start[0], start[1], 0)]  # (moves, r, c, energy)
    visited = set()

    while q:
        moves, r, c, energy = heapq.heappop(q)

        if (r, c, energy) in visited:
            continue
        visited.add((r, c, energy))

        if (r, c) == goal:
            print("Yes")
            return

        meds_at_current = []
        for mr, mc, me in medicines:
            if (mr, mc) == (r, c):
                meds_at_current.append((mr, mc, me))
        
        for mr, mc, me in meds_at_current:
            if (moves + 1, r, c, me) not in visited:
                heapq.heappush(q,(moves + 1, r, c, me))

        if energy > 0:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' :
                    if (moves + 1, nr, nc, energy - 1) not in visited:
                        heapq.heappush(q, (moves + 1, nr, nc, energy - 1))
    
    print("No")


solve()