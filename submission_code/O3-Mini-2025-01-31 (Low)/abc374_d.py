def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = float(next(it))
    T = float(next(it))
    segments = []
    for _ in range(N):
        a = float(next(it))
        b = float(next(it))
        c = float(next(it))
        d = float(next(it))
        segments.append(((a, b), (c, d)))
        
    # For each segment, we consider two possible ways of printing:
    #   (i) Move (without printing) to one endpoint then draw to the other.
    #   (ii) Alternatively start at the other endpoint.
    # We index states: for segment i, state 2*i represents the first endpoint (as given)
    # and state 2*i+1 represents the second endpoint.
    # When printing a segment, the laser always moves from the chosen starting endpoint
    # to the opposite endpoint along the segment at speed T.
    # Our cost: travel cost (at speed S) for repositioning (when not printing)
    # and printing cost (at speed T) which is the segment length / T.
    
    total_states = 2 * N  # Each segment gives 2 possible states (which endpoint is reached after printing)
    endpoints = []        # List of endpoints corresponding to states
    seg_print_cost = []   # Printing cost for each segment, same regardless of printing direction.
    
    for i in range(N):
        ep0 = segments[i][0]
        ep1 = segments[i][1]
        endpoints.append(ep0)
        endpoints.append(ep1)
        d = math.hypot(ep0[0] - ep1[0], ep0[1] - ep1[1])
        seg_print_cost.append(d / T)
        
    # dp[mask, state] represents the minimum time required after printing all segments in mask
    # and ending at the coordinate given by state (which is one of our 2*N endpoints).
    #
    # We will use bitmask dp where mask is an integer with N bits (each representing a segment).
    #
    # Initially the laser is at (0, 0). For each segment, we can choose a direction:
    #   - Go from (0,0) to one of its endpoints (cost distance/S),
    #   - Then print the segment (cost = seg_print_cost).
    #
    # After printing a segment, note that if we chose a starting endpoint (state),
    # the laser finishes at the opposite endpoint.
    
    INF = float('inf')
    dp = {}
    
    # Initialize first moves: print one segment from (0,0)
    for i in range(N):
        for choice in [0, 1]:
            # For segment i, if choice is 0 then:
            #    starting endpoint is endpoints[2*i + 0], and finishing point is endpoints[2*i + 1].
            # If choice is 1 then starting endpoint is endpoints[2*i + 1] and finishing endpoint is endpoints[2*i + 0].
            start = 2 * i + choice        # starting endpoint (the one we travel to)
            finish = 2 * i + (1 - choice)   # finishing endpoint (after printing)
            # Compute travel time from (0,0) to starting endpoint:
            sx, sy = endpoints[start]
            travel_time = math.hypot(sx, sy) / S
            total_time = travel_time + seg_print_cost[i]
            mask = 1 << i
            key = (mask, finish)
            if key not in dp or dp[key] > total_time:
                dp[key] = total_time

    # Extend solution over all remaining segments.
    # At each stage, dp[(mask, last_state)] gives us a state where we've printed
    # the segments indicated by mask and ended at endpoints[last_state].
    for printed_count in range(1, N):
        # To avoid modifying dp while iterating, use a temporary dictionary for updates.
        new_dp = {}
        for (mask, last_state), cost in dp.items():
            for j in range(N):
                if mask & (1 << j):
                    continue  # segment j already printed
                for choice in [0, 1]:
                    # For segment j: if we choose printing starting at endpoints[2*j + choice],
                    # then printing will finish at endpoints[2*j + (1 - choice)].
                    start = 2 * j + choice
                    finish = 2 * j + (1 - choice)
                    # Travel from the current position (endpoints[last_state]) to the starting endpoint.
                    cur_x, cur_y = endpoints[last_state]
                    st_x, st_y = endpoints[start]
                    travel_time = math.hypot(cur_x - st_x, cur_y - st_y) / S
                    new_cost = cost + travel_time + seg_print_cost[j]
                    new_mask = mask | (1 << j)
                    key = (new_mask, finish)
                    if key not in new_dp or new_dp[key] > new_cost:
                        new_dp[key] = new_cost
        # Merge new_dp into dp
        for key, val in new_dp.items():
            if key in dp:
                dp[key] = min(dp[key], val)
            else:
                dp[key] = val

    # The final answer is the minimum cost among states where mask has all segments printed.
    final_mask = (1 << N) - 1
    ans = INF
    for (mask, state), cost in dp.items():
        if mask == final_mask:
            ans = min(ans, cost)
    sys.stdout.write("{:.12f}".format(ans))
    
if __name__ == '__main__':
    main()