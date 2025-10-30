import sys
from collections import deque

def solve():
    # Read input
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())

    # Convert 1-indexed to 0-indexed
    start_r, start_c = A - 1, B - 1
    end_r, end_c = C - 1, D - 1

    # dist[r][c] stores minimum kicks required to reach cell (r, c)
    dist = [[float('inf')] * W for _ in range(H)]
    dist[start_r][start_c] = 0

    # Deque for 0-1 BFS
    # Cells with lower kick count (cost 0 transitions) are added to the front.
    # Cells with higher kick count (cost 1 transitions) are added to the back.
    q = deque([(start_r, start_c)])

    # Directions for movement (cost 0) - Standard 4 cardinal directions
    dr_move = [-1, 1, 0, 0]
    dc_move = [0, 0, -1, 1]

    # Directions for kick (cost 1)
    # Kick affects cells up to 2 steps in one of 4 cardinal directions
    dr_kick_template = [-1, 1, 0, 0]
    dc_kick_template = [0, 0, -1, 1]

    while q:
        # Pop cell with minimum kicks from the front.
        # This guarantees that when we process a cell (r, c), dist[r][c]
        # is the minimum number of kicks required to reach it.
        r, c = q.popleft()
        k = dist[r][c] # Current minimum kicks to reach (r, c)

        # If we reached the destination, we found the minimum kicks
        if r == end_r and c == end_c:
            print(k)
            return

        # --- Explore Cost 0 Moves ---
        # From cell (r, c), we can move to adjacent cells (nr, nc).
        # This move is a cost-0 transition in the 0-1 BFS if it allows us to reach
        # the adjacent cell (nr, nc) with the same number of kicks (k).
        # This happens if dist[nr][nc] can be improved to k.
        # This effectively explores all cells reachable from (r,c) using paths
        # where all intermediate cells are reachable with at most k kicks.
        # It represents moving between cells that are 'open' at kick level k.
        for i in range(4):
            nr, nc = r + dr_move[i], c + dc_move[i]

            # Check bounds
            if 0 <= nr < H and 0 <= nc < W:
                 # If we found a potentially better path (with the same number of kicks)
                 # Note: The original grid S[nr][nc] is NOT checked here for moves.
                 # Reachability is determined by dist values in 0-1 BFS. If we can
                 # reach (r, c) with k kicks, and (nr, nc) is adjacent, and (nr, nc)
                 # hasn't been reached with fewer kicks (dist[nr][nc] > k), then
                 # we can reach (nr, nc) with k kicks by moving from (r, c).
                 # This models movement through any cell that is 'open' at kick level k.
                 if dist[nr][nc] > k:
                    dist[nr][nc] = k
                    # Push to the front because it's a cost-0 transition
                    q.appendleft((nr, nc))

        # --- Explore Cost 1 Kicks ---
        # From cell (r, c), we can perform a kick. This costs 1 additional kick,
        # resulting in a total of k + 1 kicks.
        # This kick makes cells up to 2 steps away in the chosen direction
        # reachable with k+1 kicks.
        for i in range(4): # For each of the 4 cardinal directions
            dr, dc = dr_kick_template[i], dc_kick_template[i]
            for step in range(1, 3): # For 1 or 2 steps away in that direction
                ar, ac = r + dr * step, c + dc * step

                # Check bounds for the affected cell
                if 0 <= ar < H and 0 <= ac < W:
                    # Performing a kick from (r,c) allows us to reach (ar,ac) with k+1 kicks.
                    # We update if this is a better path (in terms of kicks).
                    # Note: We do NOT check if S[ar][ac] is '#'. Kicking a '.' doesn't change it,
                    # but reaching it via this 'kick path' might still be relevant if dist > k+1.
                    # However, kicking a '#' is the primary way to open new paths.
                    # The condition dist[ar][ac] > k+1 is sufficient for correctness
                    # in finding the minimum number of kicks.
                    if dist[ar][ac] > k + 1:
                        dist[ar][ac] = k + 1
                        # Push to the back because it's a cost-1 transition
                        q.append((ar, ac))

    # The problem guarantees a path exists, so we should always find the end.
    # This line should technically not be reached in a valid problem instance.
    # print(-1) # Fallback or error case if end is truly unreachable

solve()