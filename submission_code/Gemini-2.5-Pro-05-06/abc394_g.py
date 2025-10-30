import collections

def solve():
    H, W = map(int, input().split())
    F_heights = []
    for _ in range(H):
        F_heights.append(list(map(int, input().split())))

    Q_count = int(input()) # Q_count variable from input format
    
    results = []
    for _ in range(Q_count):
        r_start, c_start, y_start, r_target, c_target, z_target = map(int, input().split())
        # Adjust to 0-indexed
        r_start -= 1
        c_start -= 1
        r_target -= 1
        c_target -= 1

        if r_start == r_target and c_start == c_target:
            results.append(abs(y_start - z_target))
            continue

        # dist maps (row, col) tuple to another map: {floor: min_stairs}
        # For each query, dist must be fresh.
        dist = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        
        dq = collections.deque()

        dist[(r_start, c_start)][y_start] = 0
        dq.append((0, r_start, c_start, y_start)) # (stairs_count, r, c, current_floor)
        
        min_total_stairs = float('inf')
        # If already in target building, initialize min_total_stairs
        # This happens if (r_start,c_start) == (r_target,c_target) but y_start != z_target
        # (Handled by the check at the beginning of the loop)

        while dq:
            s_curr, r_curr, c_curr, f_curr = dq.popleft()

            if s_curr > dist[(r_curr, c_curr)][f_curr]: # Already found shorter path to this state
                continue
            
            # Update min_total_stairs if at target building
            if r_curr == r_target and c_curr == c_target:
                 min_total_stairs = min(min_total_stairs, s_curr + abs(f_curr - z_target))

            # Pruning: if s_curr itself is already too large
            if s_curr >= min_total_stairs:
                continue

            # Move 1: Use stairs
            # Move down
            if f_curr > 1:
                s_new = s_curr + 1
                # Pruning: if new stair count is already too high
                if s_new < min_total_stairs and s_new < dist[(r_curr, c_curr)][f_curr - 1]:
                    dist[(r_curr, c_curr)][f_curr - 1] = s_new
                    dq.append((s_new, r_curr, c_curr, f_curr - 1))
            
            # Move up
            if f_curr < F_heights[r_curr][c_curr]:
                s_new = s_curr + 1
                # Pruning
                if s_new < min_total_stairs and s_new < dist[(r_curr, c_curr)][f_curr + 1]:
                    dist[(r_curr, c_curr)][f_curr + 1] = s_new
                    dq.append((s_new, r_curr, c_curr, f_curr + 1))

            # Move 2: Use walkway
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                r_adj, c_adj = r_curr + dr, c_curr + dc

                if 0 <= r_adj < H and 0 <= c_adj < W:
                    if F_heights[r_adj][c_adj] >= f_curr: # Adjacent building tall enough
                        s_new = s_curr # No stairs used for walkway
                        # Pruning (s_new == s_curr, so this check is s_curr < min_total_stairs)
                        # This is already covered by the check at the start of the loop iteration.
                        # Still, an explicit check on s_new is fine.
                        if s_new < min_total_stairs and s_new < dist[(r_adj, c_adj)][f_curr]:
                            dist[(r_adj, c_adj)][f_curr] = s_new
                            dq.appendleft((s_new, r_adj, c_adj, f_curr))
        
        results.append(min_total_stairs)

    for res in results:
        print(res)

solve()