import sys
import heapq

def solve():
    N, M = map(int, sys.stdin.readline().split())

    train_params = [] 
    # rev_adj[B] stores list of (A, train_data_index) for trains A -> B
    # This represents incoming edges to B in the physical graph.
    rev_adj = [[] for _ in range(N + 1)] 

    for i in range(M):
        # l_i, d_i, k_i, c_i, A_i, B_i
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        # Store train parameters. Using a dictionary for clarity but tuple is fine.
        train_params.append({'l':l, 'd':d, 'k':k, 'c':c}) 
        rev_adj[B].append((A, i)) # Train from A to B means an edge A->B. For reversed graph, B to A.

    # dp[u] stores the latest time one can DEPART from station u and still reach station N.
    # Initialize with -1 (representing negative infinity / unreachable)
    dp = [-1] * (N + 1)

    # For station N, we can "depart" at a very late time conceptually.
    # Max possible departure time from a station is l_i + (k_i-1)*d_i.
    # With l_i, d_i, k_i up to 10^9, this can be roughly 10^9 + (10^9-1)*10^9 ~ 10^18.
    # L_INF must be larger than any such possible departure time, even after subtracting c_i.
    L_INF = 4 * 10**18 # Using a sufficiently large number for infinity
    dp[N] = L_INF

    # Priority queue stores (-time, station_idx) to simulate a max-heap with heapq.
    # heapq implements a min-heap, so we negate priorities (times).
    pq = [(-L_INF, N)] # (negated_departure_time_from_station, station_id)

    while pq:
        neg_current_max_dep_time_from_u, u = heapq.heappop(pq)
        current_max_dep_time_from_u = -neg_current_max_dep_time_from_u

        # If this entry is stale (we've found a better path to u already), skip it.
        if current_max_dep_time_from_u < dp[u]:
            continue

        # For station u, iterate over all trains that arrive at u.
        # These are incoming edges in the physical graph from prev_station_A to u.
        for prev_station_A, train_idx in rev_adj[u]:
            train = train_params[train_idx]
            l, d, k, c = train['l'], train['d'], train['k'], train['c']
            
            # We can depart from station u at `current_max_dep_time_from_u`.
            # To do this, we must arrive at u by `current_max_dep_time_from_u`.
            # The train from `prev_station_A` to `u` has travel time `c`.
            # So, Departure_from_A + c <= current_max_dep_time_from_u
            # Hence, Departure_from_A <= current_max_dep_time_from_u - c
            # This is T_limit: the latest time we can depart `prev_station_A` using this specific train service
            # and still make the "connection" at `u` (i.e., depart `u` by `current_max_dep_time_from_u`).
            T_limit = current_max_dep_time_from_u - c

            # If this latest allowed departure T_limit is earlier than the first train of the service (l),
            # then no train in this service can be used.
            if T_limit < l:
                continue
            
            # We need to find the latest departure from `prev_station_A` using this service
            # that is <= `T_limit`.
            # Departure times are l, l+d, ..., l+(k-1)d.
            # We need to find max m such that:
            #   1) l + m*d <= T_limit
            #   2) m <= k-1 (since there are k trains, m ranges from 0 to k-1)
            # From (1): m*d <= T_limit - l
            #           m <= floor((T_limit - l) / d)  (since d >= 1 from constraints)
            
            num_steps_m_calc = (T_limit - l) // d
            
            # m must be at most k-1 and at most num_steps_m_calc.
            # Since k >= 1, k-1 >= 0.
            # Since T_limit >= l, T_limit - l >= 0. Since d >= 1, num_steps_m_calc >= 0.
            # Thus, m_chosen will be non-negative.
            m_chosen = min(k - 1, num_steps_m_calc)
            
            candidate_dep_time_A = l + m_chosen * d
            
            # If this path offers a later departure time for `prev_station_A`, update and push to PQ.
            if candidate_dep_time_A > dp[prev_station_A]:
                dp[prev_station_A] = candidate_dep_time_A
                heapq.heappush(pq, (-dp[prev_station_A], prev_station_A))

    # Prepare output
    output_lines = []
    for i in range(1, N): # Stations 1 to N-1 as per problem statement
        if dp[i] == -1:
            output_lines.append("Unreachable")
        else:
            output_lines.append(str(dp[i]))
    
    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    solve()