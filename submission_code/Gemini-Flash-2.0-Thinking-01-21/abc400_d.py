import sys
from collections import deque

# Read input
H, W = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]
A, B, C, D = map(int, sys.stdin.readline().split())

# Convert 1-based to 0-based indexing
start_r, start_c = A - 1, B - 1
dest_r, dest_c = C - 1, D - 1

# Distance array initialized to infinity
dist = [[float('inf')] * W for _ in range(H)]

# Deque for 0-1 BFS
q = deque()

# Starting cell has 0 kicks
dist[start_r][start_c] = 0
q.appendleft((start_r, start_c))

# Directions for adjacent moves (0-cost)
adj_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Offsets for kick targets (1-cost) - cells within 2 steps in 4 cardinal directions
# Represents the *ability* to reach these locations with +1 kick from the current cell
kick_offsets = []
for dr, dc in adj_dirs:
    for k in range(1, 3):
        kick_offsets.append((dr * k, dc * k))

while q:
    r, c = q.popleft()
    d = dist[r][c]

    # If destination reached, we found the minimum
    if r == dest_r and c == dest_c:
        print(d)
        exit() # Exit immediately once destination is popped (guaranteed shortest path in 0-1 BFS)

    # Explore 1-cost moves (kicks)
    # From cell (r, c), performing a kick costs +1 kick.
    # This enables reaching any cell (kr, kc) within 2 steps in a straight line.
    # The cost to reach (kr, kc) via a kick from (r, c) is d + 1.
    # This transition represents the kick action itself.
    for dr, dc in kick_offsets:
        kr, kc = r + dr, c + dc
        if 0 <= kr < H and 0 <= kc < W:
            # If we found a shorter path to (kr, kc) using d+1 kicks
            if dist[kr][kc] > d + 1:
                dist[kr][kc] = d + 1
                q.append((kr, kc)) # Add to the back (1-cost)

    # Explore 0-cost moves (adjacent)
    # From cell (r, c) with cost d, we can move to adjacent cell (nr, nc).
    # This move is possible if (nr, nc) is currently traversable *at this cost level*.
    # The condition `dist[nr][nc] > d` implies we found a new or shorter path
    # to (nr, nc) with cost `d`. The 0-1 BFS structure ensures this path is valid.
    # The crucial point is that adjacency means a potential 0-cost move,
    # and the `dist` value check correctly prioritizes 0-cost over 1-cost paths.
    # We do NOT check `grid[nr][nc] == '.'` here because 0-cost moves can
    # occur on cells made road by previous kicks (which are accounted for by the
    # dist value update from 1-cost edges). A cell (nr, nc) reachable with cost d
    # (i.e. dist[nr][nc] <= d) is considered traversable at cost level d.
    for dr, dc in adj_dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
             # If we found a shorter path to (nr, nc) using d kicks
            if dist[nr][nc] > d:
                dist[nr][nc] = d
                q.appendleft((nr, nc)) # Add to the front (0-cost)

# The exit() inside the loop handles printing the result if the destination is reachable.
# If the loop finishes without reaching the destination (shouldn't happen based on constraints),
# the destination is unreachable.
# The problem guarantees that both start and end are road cells and reachable.