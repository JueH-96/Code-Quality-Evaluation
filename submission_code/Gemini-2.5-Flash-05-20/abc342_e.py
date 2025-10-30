import heapq
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Adjacency list for the reversed graph: stores (l, d, k, c, A) for trains arriving at B
    # adj[B] contains a list of (l_i, d_i, k_i, c_i, A_i) tuples, meaning a train
    # from station A_i to B (current_station), departing at l_i, l_i+d_i, ..., l_i+(k_i-1)d_i
    # and arriving at t+c_i.
    adj = [[] for _ in range(N + 1)]

    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        adj[B].append((l, d, k, c, A))

    # f[s] will store the maximum departure time from station s that allows reaching N.
    # Initialize f values. f[N] is conceptually infinity, as we arrive at N.
    # All train times (l_i, l_i + x*d_i) are positive integers.
    # The maximum possible departure time l_i + (k_i-1)*d_i can be up to
    # 10^9 + (10^9 - 1) * 10^9 which is roughly 10^18.
    # A path can consist of up to N-1 segments. Max accumulated travel time
    # could be (N-1) * max_c = (2*10^5 - 1) * 10^9 which is roughly 2*10^14.
    # So, intermediate f values or arrival times could slightly exceed 10^18.
    # A safe INF_TIME value would be larger than any possible valid time.
    # 4 * 10^18 should be sufficient.
    INF_TIME = 4 * 10**18 
    
    # Initialize f array: f[N] is INF_TIME, all other f[s] are -1 (representing -infinity)
    # as valid departure times are positive.
    f = [-1] * (N + 1)
    f[N] = INF_TIME

    # Priority queue stores (-max_depart_time, station).
    # We use negative max_depart_time because heapq is a min-heap,
    # and we want to retrieve the station with the largest f value.
    pq = [(-INF_TIME, N)] 

    while pq:
        neg_current_f, current_station = heapq.heappop(pq)
        current_f = -neg_current_f

        # If we have already processed this station with a higher (or equal) f value,
        # then this popped value is stale, so skip.
        if current_f < f[current_station]:
            continue

        # Iterate over all train schedules that have current_station as their destination (B_i)
        # (l, d, k, c, prev_station) corresponds to (l_i, d_i, k_i, c_i, A_i) where B_i = current_station
        for l, d, k, c, prev_station in adj[current_station]:
            # For a train departing from `prev_station` at `t` and arriving at `current_station` at `t+c`,
            # for this train to be part of the optimal path to N, its arrival time `t+c`
            # must be no later than `current_f` (the latest time one can depart from `current_station` to N).
            # So, `t+c <= current_f`, which implies `t <= current_f - c`.
            T_max_allowed_dep_time = current_f - c

            # Check if any train in this schedule (l, d, k) can be used.
            # The earliest departure time is `l`. If `T_max_allowed_dep_time` is less than `l`,
            # then even the earliest train of this schedule cannot be used.
            if T_max_allowed_dep_time < l:
                continue

            # We need to find the largest departure time `t_candidate = l + x*d`
            # such that `t_candidate <= T_max_allowed_dep_time` and `0 <= x < k`.
            # From `l + x*d <= T_max_allowed_dep_time`:
            # `x*d <= T_max_allowed_dep_time - l`
            # `x <= (T_max_allowed_dep_time - l) / d` (using integer division `//` for floor).
            
            x_upper_bound_by_time = (T_max_allowed_dep_time - l) // d
            
            # Also, x must be less than k, so x <= k-1.
            x_optimal = min(k - 1, x_upper_bound_by_time)
            
            # Calculate the actual departure time from `prev_station` for this specific optimal train
            t_candidate = l + x_optimal * d

            # If this `t_candidate` is greater than the currently known latest departure time
            # from `prev_station` (`f[prev_station]`), update it and push to PQ.
            if t_candidate > f[prev_station]:
                f[prev_station] = t_candidate
                heapq.heappush(pq, (-t_candidate, prev_station))

    # Print the results for stations 1 to N-1
    # Use sys.stdout.write for potentially faster output than print()
    for i in range(1, N):
        if f[i] == -1:
            sys.stdout.write("Unreachable
")
        else:
            sys.stdout.write(str(f[i]) + "
")

# Call the solver function
solve()