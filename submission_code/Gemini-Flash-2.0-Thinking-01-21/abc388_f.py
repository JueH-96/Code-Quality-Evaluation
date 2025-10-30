import sys
import queue

# Function to read input
def read_input():
    N, M, A, B = map(int, sys.stdin.readline().split())
    bad_intervals = []
    for _ in range(M):
        L, R = map(int, sys.stdin.readline().split())
        bad_intervals.append((L, R))
    return N, M, A, B, bad_intervals

# Function to solve the problem
def solve():
    N, M, A, B, bad_intervals = read_input()

    # Handle M=0 case separately
    if M == 0:
        if N == 1: # Constraint N>=2, but good to be safe
             print("Yes")
             return
        
        # If N > 1, check if N-1 can be formed by sum of steps >= A
        if N - 1 < A:
             print("No")
             return

        if B == A:
            if (N - 1) % A == 0:
                print("Yes")
            else:
                print("No")
        else: # B > A
            # Frobenius number for {A, A+1, ..., B} is bounded by B*(A-1) if A>1.
            # Any sum k >= T_reach is representable.
            T_reach = B * (A - 1) if A > 1 else 0

            if N - 1 >= T_reach:
                 print("Yes")
            else:
                 # Need DP for small sums up to N-1 if N-1 < T_reach
                 max_sum_dp = N - 1
                 reachable_sum = [False] * (max_sum_dp + 1)
                 reachable_sum[0] = True
                 q_sum = queue.Queue()
                 q_sum.put(0)

                 # Use a set for visited sums to avoid redundant queueing
                 visited_sum = {0}

                 while not q_sum.empty():
                     s = q_sum.get()
                     
                     if s == N - 1: # Reached the target sum
                         print("Yes")
                         return

                     if s > max_sum_dp: continue 

                     for step in range(A, B + 1):
                         next_s = s + step
                         if next_s <= max_sum_dp and next_s not in visited_sum:
                             reachable_sum[next_s] = True
                             visited_sum.add(next_s)
                             q_sum.put(next_s)

                 print("No") # N-1 is not reachable by sum of steps <= max_sum_dp
        return

    # M > 0 case
    bad_intervals.sort()

    # Calculate DP for reachable sums from 0 up to a certain threshold
    # T_reach = B*(A-1) if B > A and A > 1 else 0
    T_reach = B * (A - 1) if B > A and A > 1 else 0
    
    # The maximum relevant difference N-u in the last segment might be N-(R_M+1).
    # We need DP for sums up to min(N-(R_M+1), T_reach + B)
    max_relevant_diff_in_last_segment = N - (bad_intervals[-1][1] + 1)
    if max_relevant_diff_in_last_segment < 0: # N <= R_M, should not happen based on R_i < N
        max_sum_dp_needed = B # Or some small minimum
    else:
        max_sum_dp_needed = min(max_relevant_diff_in_last_segment, T_reach + B)
        
    # Ensure max_sum_dp is large enough for the DP itself to find sums.
    # The DP should cover sums up to T_reach or up to the max needed diff if smaller.
    # Use a safe upper bound for DP table size.
    max_sum_dp_table_size = T_reach + B + 50 # Add some margin
    max_sum_dp = min(max_relevant_diff_in_last_segment, max_sum_dp_table_size)


    reachable_sum = [False] * (max_sum_dp + 1)
    if max_sum_dp >= 0:
      reachable_sum[0] = True
    q_sum = queue.Queue()
    if 0 <= max_sum_dp:
        q_sum.put(0)

    visited_sum = {0}

    while not q_sum.empty():
        s = q_sum.get()
        
        if s > max_sum_dp: continue 

        for step in range(A, B + 1):
            next_s = s + step
            if next_s <= max_sum_dp and next_s not in visited_sum:
                reachable_sum[next_s] = True
                visited_sum.add(next_s)
                q_sum.put(next_s)


    # BFS on squares
    q = queue.Queue()
    visited = set()

    q.put(1)
    visited.add(1)

    # Determine the cap for visited squares
    # Cap = max R_i + threshold + B + margin
    max_R = bad_intervals[-1][1]
    max_bfs_square = max_R + T_reach + B + 5
    # Cap at N, we don't need to add squares > N to visited or queue
    max_bfs_square = min(N, max_bfs_square) 

    # Keep track of the current bad interval index for efficiency
    bad_idx = 0 

    while not q.empty():
        u = q.get()

        if u == N:
            print("Yes")
            return

        # Check if u is in the last segment [R_M+1, N]
        last_segment_start = bad_intervals[-1][1] + 1 
        if u >= last_segment_start:
            diff = N - u
            if diff >= 0: 
                if B > A:
                    # Check if diff is reachable by sums >= T_reach or by DP
                    if diff >= T_reach: print("Yes"); return
                    # Check against the DP table
                    if diff <= max_sum_dp and reachable_sum[diff]: print("Yes"); return
                else: # B == A
                    if diff % A == 0: print("Yes"); return

        # Explore neighbors
        for step in range(A, B + 1):
            v = u + step

            if v > N: continue # Cannot go beyond N

            # Check if v is bad.
            is_bad = False
            # Advance bad_idx if necessary to find the interval relevant to u
            while bad_idx < M and bad_intervals[bad_idx][1] < u:
                 bad_idx += 1

            # Check intervals from the current bad_idx that could potentially contain v
            check_bad_idx = bad_idx
            while check_bad_idx < M and bad_intervals[check_bad_idx][0] <= v:
                 if v <= bad_intervals[check_bad_idx][1]:
                      is_bad = True
                      break
                 check_bad_idx += 1

            if is_bad: continue

            # Add to queue only if within limit or is N
            if v <= max_bfs_square or v == N:
                 if v not in visited:
                    visited.add(v)
                    q.put(v)

    # If BFS finishes without reaching N
    print("No")


# Execute the solver
solve()