import sys
from collections import deque
from bisect import bisect_left

# Increase recursion depth for safety, although SPFA is iterative
# sys.setrecursionlimit(300000) # Max N, M = 2*10^5

def solve():
    # Read input
    # N: number of cities, M: number of trains, X1: given delay for train 1
    N, M, X1 = map(int, sys.stdin.readline().split())

    # Store train data as 0-indexed (A, B, S, T)
    # Problem uses 1-indexed trains 1..M. Train 1 is index 0 internally.
    trains_data = []
    for i in range(M):
        a, b, s, t = map(int, sys.stdin.readline().split())
        trains_data.append((a, b, s, t))

    # Group trains by departure city and arrival city for efficient neighbor finding
    # Store relevant info (Time, OtherTime, train_idx) for sorting
    # For departing trains, sort by S. For arriving trains, sort by T.
    departing_trains_by_city = {}
    arriving_trains_by_city = {}

    for i in range(M):
        a, b, s, t = trains_data[i]
        if a not in departing_trains_by_city:
            departing_trains_by_city[a] = []
        # Store (S, T, train_idx) for departing trains
        departing_trains_by_city[a].append((s, t, i))

        if b not in arriving_trains_by_city:
            arriving_trains_by_city[b] = []
        # Store (T, S, train_idx) for arriving trains
        arriving_trains_by_city[b].append((t, s, i))

    # Sort trains within each city group
    # Sorting by the first element of the tuple is the default
    for city in departing_trains_by_city:
        departing_trains_by_city[city].sort() # Sort by S
    for city in arriving_trains_by_city:
        arriving_trains_by_city[city].sort() # Sort by T


    # d[i] will store the minimum required delay X_{i+1} for train i+1 (0-indexed i)
    # Problem statement refers to X_1..X_M. Train 1 is index 0.
    # So d[0] corresponds to X_1, d[1] to X_2, ..., d[M-1] to X_M.
    d = [0] * M
    d[0] = X1 # Train 1 (index 0) has initial required delay X1

    # SPFA initialization
    Q = deque()
    in_queue = [False] * M

    # Initial nodes to process are all trains.
    # Their initial delays are set (d[0] = X1, d[k] = 0 for k >= 1).
    # These initial values are lower bounds. Add all nodes to the queue initially.
    for i in range(M):
        Q.append(i)
        in_queue[i] = True

    # SPFA loop
    # Propagate minimum required delays. d[k] values are non-decreasing because constraints are lower bounds.
    while Q:
        u_idx = Q.popleft()
        in_queue[u_idx] = False

        current_du = d[u_idx]

        # Find neighbors v of u where u -> v exists in the dependency graph.
        # u -> v means train u arrives at city B_u, train v departs from city A_v,
        # B_u = A_v and T_u <= S_v.
        # So, v is a train departing from city B_u.

        city_Bu = trains_data[u_idx][1] # B_u
        time_Tu = trains_data[u_idx][3] # T_u

        # Iterate through trains j_idx departing from city_Bu
        # We only care about trains j (1-indexed) where A_j = city_Bu.
        # These are available in `departing_trains_by_city[city_Bu]`, sorted by S_j.

        # Get the list of trains departing from the arrival city of train u
        departing_list = departing_trains_by_city.get(city_Bu, [])

        # Use bisect_left to find the index of the first train j in departing_list
        # where S_j >= time_Tu. `departing_list` is sorted by S.
        # Elements are (S_j, T_j, j_idx). Bisect on the first element S_j.
        # The key (time_Tu, -1, -1) works because bisect_left compares elements lexicographically.
        # Using large negative values for dummy S/T might be safer if times can be negative,
        # but problem states 0 <= S_i < T_i. Using -1 is fine.
        start_idx = bisect_left(departing_list, (time_Tu, -1, -1))

        # Iterate through relevant departing trains j
        # These are the trains j where a transfer from train u is originally possible (T_u <= S_j)
        for item in departing_list[start_idx:]:
            S_j, T_j_dummy, j_idx = item # j_idx is 0-indexed train index

            # Edge exists from u_idx to j_idx in dependency graph with weight T_u - S_j
            weight = time_Tu - S_j # This is <= 0

            # Relaxation: d[j_idx] >= d[u_idx] + weight
            # The new minimum required delay for train j is at least the current delay of train u + the required minimum difference.
            new_dj = current_du + weight

            # Apply non-negativity constraint for trains 2..M (indices 1..M-1)
            # Train 1 is index 0. Its delay d[0] is fixed at X1.
            if j_idx >= 1: # corresponds to train 2 or later
                 # Apply non-negativity: minimum delay must be 0
                 new_dj = max(new_dj, 0)

                 # Update d[j_idx] if the new minimum required delay (after considering all constraints) is larger
                 if new_dj > d[j_idx]:
                      d[j_idx] = new_dj
                      # If d[j_idx] was updated and not already in queue, add it
                      if not in_queue[j_idx]:
                          Q.append(j_idx)
                          in_queue[j_idx] = True
            # else j_idx == 0 (Train 1).
            # d[0] is fixed at X1. Constraints like X_1 >= X_u + T_u - S_1
            # provide upper bounds on X_u (X_u <= X_1 - (T_u - S_1)), which do not force d[0] to increase.
            # Since we are minimizing delays (finding minimum required values, which are lower bounds),
            # upper bound constraints on *other* variables do not affect the minimum lower bound of the current variable.
            # So, we do not update d[0] here.

    # The result is X_2, ..., X_M, which are d[1], ..., d[M-1]
    # The algorithm ensures d[k] >= 0 for k >= 1 (trains 2..M).
    # d[0] is X1 which is positive.
    output = d[1:]

    print(*output)

# Read from standard input and run the solve function
solve()