import sys
from collections import deque

def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    # Convert 1-based to 0-based indexing
    start_r, start_c = A - 1, B - 1
    end_r, end_c = C - 1, D - 1

    # dist[r][c] = minimum kicks performed on a path from start to (r, c)
    dist = [[float('inf')] * W for _ in range(H)]
    dist[start_r][start_c] = 0

    # Deque for 0-1 BFS
    # Elements are (row, col)
    deque_q = deque([(start_r, start_c)])

    # Directions for movement (walk) - Adjacent cells
    dr_walk = [-1, 1, 0, 0]
    dc_walk = [0, 0, -1, 1]

    # Directions for kick - Up, Down, Left, Right
    dr_kick_dir = [-1, 1, 0, 0]
    dc_kick_dir = [0, 0, -1, 1]

    while deque_q:
        r, c = deque_q.popleft()
        d = dist[r][c]

        # If we reached the destination, we can stop and print the result
        if r == end_r and c == end_c:
            print(d)
            return

        # Explore 0-cost moves (Walking to adjacent cells)
        # From cell (r, c), we can potentially move to adjacent cell (nr, nc).
        # This is a 0-cost move if it leads to a shorter path to (nr, nc)
        # with the same number of kicks. The 0-1 BFS structure implicitly
        # handles the "road" property by processing nodes in increasing order of kicks.
        # If dist[nr][nc] > d, it means we found a path to (nr, nc) using d kicks
        # (by walking from (r, c)), which is better than its current distance.
        # This implies (nr, nc) is now accessible with d kicks.
        for i in range(4):
            nr, nc = r + dr_walk[i], c + dc_walk[i]
            if 0 <= nr < H and 0 <= nc < W:
                if dist[nr][nc] > d:
                    dist[nr][nc] = d
                    deque_q.appendleft((nr, nc)) # Add to front (0-cost)

        # Explore 1-cost moves (Kicking from (r, c))
        # Performing a kick from cell (r, c) costs 1 kick. This action makes
        # cells at most 2 steps away in a chosen direction traversable.
        # The minimum kicks required to reach such an affected cell (nr, nc)
        # via this kick is d + 1.
        for i in range(4): # Directions: Up, Down, Left, Right
            for step in [1, 2]: # 1 or 2 steps away
                nr, nc = r + dr_kick_dir[i] * step, c + dc_kick_dir[i]
                if 0 <= nr < H and 0 <= nc < W:
                    # We can reach cell (nr, nc) with d+1 kicks by performing a kick from (r, c).
                    # If this is better than the current recorded distance to (nr, nc), update it.
                    if dist[nr][nc] > d + 1:
                        dist[nr][nc] = d + 1
                        deque_q.append((nr, nc)) # Add to back (1-cost)

    # The loop will terminate once the destination is reached because of the check inside.
    # If the destination is unreachable, the loop finishes and dist[end_r][end_c] remains inf.
    # Problem constraints guarantee reachability.

solve()