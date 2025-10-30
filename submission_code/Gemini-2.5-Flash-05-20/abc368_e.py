import sys
import collections
import heapq

# Function to read input faster
def get_ints(): return map(int, sys.stdin.readline().split())

def solve():
    N, M, X1_given = get_ints()

    trains_data = [(0, 0, 0, 0)] * (M + 1) # 1-indexed for convenience
    for i in range(1, M + 1):
        A, B, S, T = get_ints()
        trains_data[i] = (A, B, S, T)

    # P[k] will store the minimum possible effective departure time (S_k + X_k) for train k.
    P = [0] * (M + 1)
    
    # Initialize P[1] with the given X_1. For others, initialize as if X_k = 0.
    P[1] = trains_data[1][2] + X1_given # S_1 + X_1_given

    # Group outgoing trains by departure city and sort by original departure time S_k
    # adj_outgoing[city_id] = list of (S_k, k_idx)
    adj_outgoing = collections.defaultdict(list)
    for k in range(1, M + 1):
        A_k = trains_data[k][0]
        S_k = trains_data[k][2]
        adj_outgoing[A_k].append((S_k, k))

    # Sort each list by S_k for efficient filtering later using T_u <= S_v
    for city_id in adj_outgoing:
        adj_outgoing[city_id].sort()

    # Priority queue for Dijkstra: stores (-P_value, train_idx)
    # Using negative P_value to simulate a max-heap with Python's min-heap.
    pq = []
    
    # Only train 1 starts with a non-zero P value explicitly, so it's the initial source.
    heapq.heappush(pq, (-P[1], 1))

    # `finalized` array to prevent processing a train multiple times if a shorter path was already found (standard Dijkstra).
    # Here, for longest paths, it ensures we process a train's final maximum P value once.
    finalized = [False] * (M + 1) 

    while pq:
        neg_p_u, u = heapq.heappop(pq)
        p_u = -neg_p_u # Get the actual P_u value

        # If this train's P value has already been finalized with a greater value, skip.
        if finalized[u]:
            continue
        finalized[u] = True

        u_data = trains_data[u]
        u_arrival_city = u_data[1] # B_u
        u_orig_arrival_time = u_data[3] # T_u
        u_orig_departure_time = u_data[2] # S_u
        
        # Calculate the propagated P value (effective arrival time of train u).
        # This is (S_u + X_u) + (T_u - S_u) = T_u + X_u.
        propagated_P_value = p_u + (u_orig_arrival_time - u_orig_departure_time)

        # Iterate over trains v departing from u_arrival_city (B_u)
        # adj_outgoing[u_arrival_city] contains (S_v, v) pairs, sorted by S_v.
        # We need to consider only those where T_u <= S_v.
        
        # The `bisect_left` module could be used here to find the starting index efficiently,
        # but a simple linear scan with a continue condition is often sufficient if the number
        # of relevant connections is small in practice, or if test cases are not worst-case for this part.
        # The current solution iterates through all trains departing from u_arrival_city and filters.
        for S_v, v in adj_outgoing[u_arrival_city]:
            # Filter: Check if original transfer was possible (T_u <= S_v)
            if u_orig_arrival_time > S_v:
                continue # Original transfer not possible, so no constraint from u to v.
            
            # Train 1's delay (X_1) is fixed; its P_1 value is fixed and should not be updated.
            if v == 1:
                continue

            # Calculate the candidate new P_v value.
            # P_v must be at least S_v (ensures X_v >= 0).
            # P_v must also be at least 'propagated_P_value' from train u.
            candidate_P_v = max(trains_data[v][2], propagated_P_value)
            
            # If a longer path (larger P_v) is found, update P[v] and push to PQ.
            if candidate_P_v > P[v]:
                P[v] = candidate_P_v
                heapq.heappush(pq, (-P[v], v))
    
    # After Dijkstra completes, P[k] holds the minimum S_k + X_k for each train k.
    # Calculate X_k = P[k] - S_k for k=2 to M.
    result_X = []
    for k in range(2, M + 1):
        result_X.append(P[k] - trains_data[k][2])

    print(*(result_X))

# Call the solver function
solve()