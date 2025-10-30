import collections
import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    
    A, B, C, D = map(int, sys.stdin.readline().split())
    
    # Adjust to 0-indexed coordinates
    start_r, start_c = A - 1, B - 1
    target_r, target_c = C - 1, D - 1
    
    # dist[r][c] will store the minimum number of kicks required to make cell (r, c) traversable.
    # Initialize with infinity.
    dist = [[float('inf')] * W for _ in range(H)]
    
    # Deque for 0-1 BFS:
    # Elements are (kicks, r, c)
    # Pushing to front (appendleft) for 0-cost edges, to back (append) for 1-cost edges.
    q = collections.deque()
    
    # Starting cell requires 0 kicks to become traversable.
    dist[start_r][start_c] = 0
    q.append((0, start_r, start_c))
    
    # Directions for 0-cost moves (adjacent cells: up, down, left, right)
    dr_adj = [-1, 1, 0, 0]
    dc_adj = [0, 0, -1, 1]
    
    # Offsets for 1-cost moves (cells affected by a front kick from (r,c))
    # These are cells at most 2 steps away in one of the 4 cardinal directions.
    kick_offsets = [
        (-1, 0), (-2, 0),  # Up direction: (r-1,c), (r-2,c)
        (1, 0), (2, 0),    # Down direction: (r+1,c), (r+2,c)
        (0, -1), (0, -2),  # Left direction: (r,c-1), (r,c-2)
        (0, 1), (0, 2)     # Right direction: (r,c+1), (r,c+2)
    ]
    
    while q:
        kicks, r, c = q.popleft()
        
        # If we already found a shorter or equal path to (r, c), skip
        # (This check is crucial for correctness in 0-1 BFS and for efficiency)
        if kicks > dist[r][c]:
            continue
            
        # If we reached the target cell, this is the minimum number of kicks.
        if r == target_r and c == target_c:
            print(kicks)
            return

        # --- Explore 0-cost moves (moving to adjacent traversable cells) ---
        # From cell (r, c) (which is traversable with `kicks` cost),
        # try to move to any adjacent cell (nr, nc).
        # This move costs 0 additional kicks.
        for i in range(4):
            nr, nc = r + dr_adj[i], c + dc_adj[i]
            
            # Check if (nr, nc) is within grid bounds
            if 0 <= nr < H and 0 <= nc < W:
                # If reaching (nr, nc) with `kicks` is better than its current recorded minimum,
                # update dist and add to the front of the deque.
                # This ensures 0-cost paths are explored first.
                if kicks < dist[nr][nc]:
                    dist[nr][nc] = kicks
                    q.appendleft((kicks, nr, nc))
        
        # --- Explore 1-cost moves (performing a front kick) ---
        # From cell (r, c), perform a front kick. This costs 1 additional kick.
        # A kick makes certain cells (which might have been walls) traversable.
        for kdr, kdc in kick_offsets:
            nr_kick, nc_kick = r + kdr, c + kdc
            
            # Check if (nr_kick, nc_kick) is within grid bounds
            if 0 <= nr_kick < H and 0 <= nc_kick < W:
                # If reaching (nr_kick, nc_kick) with `kicks + 1` is better than its current recorded minimum,
                # update dist and add to the back of the deque.
                # This makes it a 1-cost edge.
                if kicks + 1 < dist[nr_kick][nc_kick]:
                    dist[nr_kick][nc_kick] = kicks + 1
                    q.append((kicks + 1, nr_kick, nc_kick))

# Execute the solver function
solve()