import sys
import sys
import sys
from collections import deque

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    N = int(sys.stdin.readline())
    medicines = {}
    for _ in range(N):
        R, C, E = map(int, sys.stdin.readline().split())
        medicines[(R-1, C-1)] = E
    # Find S and T
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start = (r, c)
            if grid[r][c] == 'T':
                target = (r, c)
    from heapq import heappush, heappop
    # Use max heap, so push negative energy
    heap = []
    heappush(heap, (-0, start[0], start[1], 0))  # energy, r, c, used_here
    max_energy = [[[-1]*2 for _ in range(W)] for __ in range(H)]
    while heap:
        neg_energy, r, c, used = heappop(heap)
        energy = -neg_energy
        if (r, c) == target:
            print("Yes")
            return
        if energy <= max_energy[r][c][used]:
            continue
        max_energy[r][c][used] = energy
        # Option to use medicine
        if (r, c) in medicines and used == 0:
            new_energy = medicines[(r, c)]
            if new_energy > max_energy[r][c][1]:
                heappush(heap, (-new_energy, r, c, 1))
        # Move to adjacent cells
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                if energy >=1 and energy -1 > max_energy[nr][nc][0]:
                    heappush(heap, (-(energy-1), nr, nc, 0))
                if (nr, nc) in medicines and max_energy[nr][nc][0] < energy -1:
                    heappush(heap, (-(energy-1), nr, nc, 0))
    print("No")

if __name__ == "__main__":
    main()