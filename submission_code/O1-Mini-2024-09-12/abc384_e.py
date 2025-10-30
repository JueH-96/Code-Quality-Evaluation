import sys
import heapq

def solve():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    H, W, X = map(int, sys.stdin.readline().split())
    P, Q = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, sys.stdin.readline().split())))
    # 0-based indexing
    P -=1
    Q -=1
    current_strength = grid[P][Q]
    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True
    heap = []
    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    for dx, dy in dirs:
        ni, nj = P + dx, Q + dy
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            heapq.heappush(heap, (grid[ni][nj], ni, nj))
    while heap:
        s, i, j = heapq.heappop(heap)
        if visited[i][j]:
            continue
        if s < current_strength / X:
            # Absorb it
            current_strength += s
            visited[i][j] = True
            # Add new neighbors to heap
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                    heapq.heappush(heap, (grid[ni][nj], ni, nj))
        else:
            # Cannot absorb, continue
            continue
    print(current_strength)