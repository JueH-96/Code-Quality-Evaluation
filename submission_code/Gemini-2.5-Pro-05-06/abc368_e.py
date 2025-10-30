import sys
import collections

# It's Python, so efficiency matters. Using sys.stdin.readline.
input = sys.stdin.readline

def main():
    N, M, X1_val = map(int, input().split())
    
    # Store train information. Using 0-indexed cities and trains.
    # train_info[i] = {'A': departure_city, 'B': arrival_city, 'S': departure_time, 'T': arrival_time}
    train_info = []
    for _ in range(M):
        A, B, S, T = map(int, input().split())
        train_info.append({'A': A - 1, 'B': B - 1, 'S': S, 'T': T})

    # X_values stores the delay for each train.
    X_values = [-1] * M 
    X_values[0] = X1_val # Train 0 (original train 1) has delay X1_val
    for i in range(1, M):
        X_values[i] = 0 # Other trains initially have 0 delay

    # adj_out[city_idx] = list of (S_j, train_idx_j) for trains j departing from city_idx
    # Sorted by S_j to allow binary search.
    adj_out = [[] for _ in range(N)]
    for i in range(M):
        adj_out[train_info[i]['A']].append((train_info[i]['S'], i))

    for city_idx in range(N):
        adj_out[city_idx].sort()

    # SPFA algorithm
    queue = collections.deque()
    in_queue = [False] * M

    # Train 0's delay is fixed and given. It can affect other trains.
    # Add train 0 to queue to start propagation.
    # Note: X1_val is positive as per problem statement.
    queue.append(0)
    in_queue[0] = True
    
    while queue:
        u = queue.popleft() # Current train index
        in_queue[u] = False

        u_arrival_city = train_info[u]['B']
        u_original_arrival_time = train_info[u]['T']
        
        # Trains departing from u_arrival_city
        departing_trains_at_city = adj_out[u_arrival_city]
        
        if not departing_trains_at_city:
            continue

        # Binary search (bisect_left logic) to find the first train v
        # such that S_v >= u_original_arrival_time
        # This finds index `idx` in `departing_trains_at_city`.
        # All trains from `departing_trains_at_city[idx:]` satisfy S_v >= T_u.
        low = 0
        high = len(departing_trains_at_city)
        start_processing_idx = len(departing_trains_at_city) # Default if all S_v < T_u

        while low < high:
            mid = (low + high) // 2
            if departing_trains_at_city[mid][0] >= u_original_arrival_time:
                start_processing_idx = mid
                high = mid
            else:
                low = mid + 1
        
        # Iterate over relevant departing trains
        for k in range(start_processing_idx, len(departing_trains_at_city)):
            v_original_departure_time, v = departing_trains_at_city[k]
            
            # X_values[0] (for train 1) is fixed and cannot be changed.
            if v == 0:  
                continue

            # Candidate X_v value based on X_u
            # X_v >= X_u + T_u_orig - S_v_orig
            candidate_X_v = X_values[u] + u_original_arrival_time - v_original_departure_time
            
            # If this candidate improves X_v (makes it larger)
            # and X_v must be non-negative. Since X_values are initialized to 0
            # and only increase, they will remain non-negative if candidate_X_v is non-negative.
            # If candidate_X_v is negative, X_values[v] (which is >=0) will not be updated by a smaller value.
            if candidate_X_v > X_values[v]:
                X_values[v] = candidate_X_v
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
    
    # Print results for trains 2 to M (0-indexed: 1 to M-1)
    print(*(X_values[k] for k in range(1, M)))

if __name__ == '__main__':
    main()