def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = float(next(it))
    T = float(next(it))
    segments = []
    for i in range(N):
        A = float(next(it)); B = float(next(it)); C = float(next(it)); D = float(next(it))
        segments.append(((A, B), (C, D)))
    
    # We'll solve the problem with a bitmask DP.
    # For each segment i we have two ways to print it:
    #   Orientation 0: printing from segments[i][0] to segments[i][1]
    #   Orientation 1: printing from segments[i][1] to segments[i][0]
    # When printing a segment, the cost consists of two parts:
    #   1. Moving (with motor speed S) from the current position to the start endpoint.
    #   2. Printing (with laser speed T) the line segment from start to finish.
    # We start from (0,0) and need to print all segments.
    # We allow any order of segments. N is small (<=6) and we have 2^N* N! possibilities.
    
    # We'll setup a DP state defined as:
    #   dp[mask, i, o] = minimal time needed after having printed the segments in 'mask',
    #     finishing with segment i printed in orientation o.
    # The definition for orientation:
    #   orientation 0 means we printed segment i from segments[i][0] -> segments[i][1],
    #   orientation 1 means we printed segment i from segments[i][1] -> segments[i][0].
    # The transition from one printed segment to the next is made by:
    #   (1) moving (no printing, speed = S) from the finish point of the last printed segment,
    #   (2) repositioning to the starting endpoint of the new segment, and then
    #   (3) printing that segment (speed = T).
    
    INF = float("inf")
    dp = {}
    
    # Start state: No segment has been printed and the laser is at (0,0).
    # So we consider printing any segment as the first operation.
    init_pos = (0.0, 0.0)
    for i in range(N):
        p0, p1 = segments[i]
        # Option 0: print from p0 to p1.
        move_time = math.hypot(p0[0] - init_pos[0], p0[1] - init_pos[1]) / S
        print_time = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / T
        dp[(1 << i, i, 0)] = move_time + print_time
        # Option 1: print from p1 to p0.
        move_time = math.hypot(p1[0] - init_pos[0], p1[1] - init_pos[1]) / S
        print_time = math.hypot(p0[0] - p1[0], p0[1] - p1[1]) / T
        dp[(1 << i, i, 1)] = move_time + print_time

    full_mask = (1 << N) - 1
    best = INF
    
    # Iterate over all DP states.
    # Our key is a tuple (mask, i, o) where mask is the bitmask of printed segments.
    for mask in range(1 << N):
        # We iterate over a static list of keys to avoid modification during iteration.
        keys = [key for key in dp if key[0] == mask]
        for state in keys:
            curr_mask, seg_idx, orient = state
            cur_time = dp[state]
            
            # Determine the current laser position: end of the last printed segment.
            if orient == 0:
                curr_pos = segments[seg_idx][1]
            else:
                curr_pos = segments[seg_idx][0]
            
            # If we have printed all segments, update the overall best.
            if curr_mask == full_mask:
                best = min(best, cur_time)
                continue
            
            # Try to print a not-yet-printed segment.
            for j in range(N):
                if curr_mask & (1 << j):
                    continue
                p0, p1 = segments[j]
                # Option 0: print from p0 to p1.
                travel = math.hypot(p0[0] - curr_pos[0], p0[1] - curr_pos[1]) / S
                seg_print = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / T
                new_cost = cur_time + travel + seg_print
                new_state = (curr_mask | (1 << j), j, 0)
                if new_state not in dp or dp[new_state] > new_cost:
                    dp[new_state] = new_cost

                # Option 1: print from p1 to p0.
                travel = math.hypot(p1[0] - curr_pos[0], p1[1] - curr_pos[1]) / S
                seg_print = math.hypot(p0[0] - p1[0], p0[1] - p1[1]) / T
                new_cost = cur_time + travel + seg_print
                new_state = (curr_mask | (1 << j), j, 1)
                if new_state not in dp or dp[new_state] > new_cost:
                    dp[new_state] = new_cost
                    
    # Output the best time found with precision acceptable.
    sys.stdout.write("{:.12f}".format(best))

if __name__ == '__main__':
    main()