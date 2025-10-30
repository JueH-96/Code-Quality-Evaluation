import heapq
import sys

# Increase recursion depth is generally not needed for iterative Dijkstra-like algorithms
# sys.setrecursionlimit(200005) 

def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())
    
    # Store train schedules in reverse graph format
    # rev_adj[v] stores list of (u, l, d, k, c) for trains from u to v
    rev_adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        # Store parameters for train from A to B
        # In reverse graph from B to A, store (A, l, d, k, c)
        rev_adj[B].append((A, l, d, k, c))

    # L[i] will store the latest departure time from station i to reach station N
    # Initialize L[N] to a very large value, others to -1 (representing -inf)
    # Use a large constant for the initial value of L[N]
    # Needs to be larger than any possible arrival time at N via a direct train
    # Max possible arrival time is roughly 10^9 + (10^9 - 1) * 10^9 + 10^9 ≈ 10^18 + 10^9
    # Setting L[N] comfortably larger, e.g., 2 * 10**18, ensures that for any direct train A_i -> N,
    # the condition L[N] - c_i >= l_i will hold, allowing the latest possible train from that schedule to be considered.
    INF_TIME = 2 * (10**18) 
    
    L = [-1] * (N + 1)
    L[N] = INF_TIME

    # Priority queue stores (negative_time, station) to extract max time
    # We want to extract the maximum time, so store negative time
    pq = []
    # Push the destination station N with its initial large latest departure time
    heapq.heappush(pq, (-L[N], N))

    # Use a set to keep track of stations already in the priority queue
    # This helps avoid redundant pushes but is not strictly necessary for correctness
    # visited_in_pq = set()
    # visited_in_pq.add(N)

    while pq:
        # Extract the station u with the current largest L[u] value
        # Since we store negative times, heapq.heappop gives the minimum negative time,
        # which corresponds to the maximum actual time.
        current_latest_at_u_neg, u = heapq.heappop(pq)
        current_latest_at_u = -current_latest_at_u_neg # Get the actual time

        # If we have already found a better (later) latest departure time for u, skip
        # This check is important because a station might be pushed to the PQ multiple times
        if current_latest_at_u < L[u]:
            continue

        # Process reverse edges from u
        # For each train schedule (l, d, k, c) from v to u
        # (v, l, d, k, c) is stored in rev_adj[u]
        for v, l, d, k, c in rev_adj[u]:
            # We need to find the latest departure time from v (t_dep)
            # such that the arrival at u (t_dep + c) is early enough to connect to a path from u to N.
            # The latest time one can arrive at u and still be able to depart from u at time L[u] to reach N
            # implies the arrival time at u must be less than or equal to L[u].
            # t_dep + c <= L[u]
            # t_dep <= L[u] - c
            
            T_limit = L[u] - c

            # Departure times from v using this schedule are l, l+d, ..., l+(k-1)d
            # We need to find the largest index j in [0, k-1] such that l + j*d <= T_limit
            
            if T_limit < l:
                # Even the earliest train departure from v (at time l) is too late to meet the arrival constraint at u
                # Required arrival time at u is T_limit + c = L[u].
                # Earliest train departs at l, arrives at l+c. If l+c > L[u], i.e., l > L[u]-c = T_limit, no train works.
                latest_dep_v = -1 # Represents -inf, meaning this schedule cannot be used to reach N via u.
            else:
                # T_limit >= l. There exist potential departure times <= T_limit starting from l.
                # We need to find the largest j such that l + j*d <= T_limit.
                # j*d <= T_limit - l
                # j <= (T_limit - l) / d
                j_cand = (T_limit - l) // d # Integer division gives floor

                # Valid indices j must also be within the range [0, k-1].
                # The largest valid index j is the minimum of k-1 and j_cand.
                j_star = min(k - 1, j_cand)
                
                # Since T_limit >= l and d > 0, (T_limit - l) >= 0, so j_cand >= 0.
                # Since k >= 1, k-1 >= 0. Thus j_star >= 0.
                # The latest departure time from v using this schedule is l + j_star * d.
                latest_dep_v = l + j_star * d

            # If this calculated latest departure time from v is greater than the current best L[v]
            if latest_dep_v > L[v]:
                L[v] = latest_dep_v
                # Push the updated value into the priority queue
                heapq.heappush(pq, (-L[v], v))

    # Output results for stations 1 to N-1
    for i in range(1, N):
        if L[i] == -1:
            print("Unreachable")
        else:
            print(L[i])

# Read input and run the solver
solve()