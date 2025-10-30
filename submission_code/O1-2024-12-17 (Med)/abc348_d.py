def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    data_pos = 2
    grid = input_data[data_pos : data_pos + H]
    data_pos += H

    # Read N and medicine info
    N = int(input_data[data_pos]); data_pos += 1
    meds = {}
    # We'll store meds as a dictionary: meds[(r, c)] = E
    for _ in range(N):
        rr = int(input_data[data_pos]) - 1
        cc = int(input_data[data_pos+1]) - 1
        E  = int(input_data[data_pos+2])
        data_pos += 3
        meds[(rr, cc)] = E

    # Locate S and T
    sr = sc = tr = tc = -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                sr, sc = r, c
            elif grid[r][c] == 'T':
                tr, tc = r, c

    # If S == T, we can immediately say "Yes"
    if sr == tr and sc == tc:
        print("Yes")
        return

    # If the start has no medicine and is not T, we begin with 0 energy and cannot move
    # However, we might still be able to use a medicine at S (if it exists) to get started.

    # dp[r][c] = the maximum energy we can have upon *arriving* at (r, c).
    # Initialize with -1 (unreachable).
    dp = [[-1]*W for _ in range(H)]
    dp[sr][sc] = 0

    # Priority queue (max-heap) will store tuples (-energy, r, c),
    # so that we always pop the largest energy state first.
    pq = []
    heapq.heappush(pq, (0, sr, sc))  # store negative of dp for a max-heap in Python

    # Directions for movements
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    # Obstacle check
    def is_obstacle(r, c):
        return grid[r][c] == '#'

    while pq:
        neg_e, r, c = heapq.heappop(pq)
        e = -neg_e  # recover the actual energy

        # If this is stale (i.e., we have a better energy in dp[r][c]), skip
        if dp[r][c] != e:
            continue

        # If we've reached T with some energy, we are done
        if r == tr and c == tc:
            print("Yes")
            return

        # Possibly "use" the medicine in (r, c) if it increases our energy
        if (r, c) in meds:
            m = meds[(r, c)]
            if m > e:
                # Update dp[r][c] if this increases the energy
                dp[r][c] = m
                e = m
                # Push again with the updated energy
                heapq.heappush(pq, (-e, r, c))
                # Do not expand neighbors now; wait for re-pop to avoid confusion
                continue

        # Now try to move to neighbors using (e) energy
        # We can only move if e > 0 (1 step costs 1 energy)
        if e > 0:
            new_e = e - 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if not is_obstacle(nr, nc):
                        if dp[nr][nc] < new_e:
                            dp[nr][nc] = new_e
                            heapq.heappush(pq, (-new_e, nr, nc))

    # If we exhaust the queue without reaching T, answer No
    print("No")