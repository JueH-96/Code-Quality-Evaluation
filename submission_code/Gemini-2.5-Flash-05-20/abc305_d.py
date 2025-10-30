import sys
from bisect import bisect_left

def solve():
    # Read N
    N = int(sys.stdin.readline())
    # Read A as a list of integers
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate prefix sums of sleep duration.
    # prefix_sleep[i] will store the total sleep time from A[0] up to A[i].
    # In 0-indexed A:
    # A[0] is A_1 (awake)
    # A[1] is A_2 (bed)
    # A[2] is A_3 (awake)
    # ...
    # Sleep intervals are [A[2k-1], A[2k]) for k=1, 2, ...
    # This means indices 1, 3, 5, ... are 'bed' times.
    # Indices 0, 2, 4, ... are 'awake' times.
    
    prefix_sleep = [0] * N
    for i in range(1, N):
        # If 'i' is an even index (0-indexed), A[i] is an "awake" time (A_3, A_5, ...).
        # This implies that the previous point A[i-1] was a "went to bed" time (A_2, A_4, ...).
        # Therefore, the interval (A[i-1], A[i]) is a sleep interval.
        if i % 2 == 0: 
            prefix_sleep[i] = prefix_sleep[i-1] + (A[i] - A[i-1])
        # If 'i' is an odd index (0-indexed), A[i] is a "went to bed" time (A_2, A_4, ...).
        # This implies that the previous point A[i-1] was an "awake" time (A_1, A_3, ...).
        # Therefore, the interval (A[i-1], A[i]) is an awake interval.
        else:
            prefix_sleep[i] = prefix_sleep[i-1]

    # Helper function to get total sleep time from A[0] up to a given 'time_val'.
    def get_total_sleep_up_to(time_val):
        # If time_val is 0, no sleep has occurred yet.
        if time_val == 0:
            return 0
        
        # Use bisect_left to find the insertion point for time_val in A.
        # This returns an index `idx` such that all elements A[0]...A[idx-1] are < time_val,
        # and all elements A[idx]...A[N-1] are >= time_val.
        # Effectively, A[idx-1] is the largest time point in A less than or equal to time_val,
        # or A[idx] is the smallest time point in A greater than or equal to time_val.
        idx = bisect_left(A, time_val)
        
        # Total sleep accumulated fully up to A[idx-1].
        # Since time_val > 0, idx will always be at least 1, so idx-1 is a valid index.
        current_sleep = prefix_sleep[idx-1]

        # Consider the interval [A[idx-1], time_val).
        # If (idx-1) is an odd index (0-indexed), it means A[idx-1] is a "went to bed" time (A_2, A_4, ...).
        # In this case, the interval (A[idx-1], A[idx]) is a sleep interval.
        if (idx - 1) % 2 == 1: 
            # Add the sleep duration within this partial sleep interval.
            current_sleep += (time_val - A[idx-1])
            
        return current_sleep

    # Read Q, the number of queries
    Q = int(sys.stdin.readline())
    results = []
    for _ in range(Q):
        l, r = map(int, sys.stdin.readline().split())
        
        # The total sleep time in the range [l, r] is calculated as
        # (total sleep up to r) - (total sleep up to l).
        sleep_in_range = get_total_sleep_up_to(r) - get_total_sleep_up_to(l)
        results.append(str(sleep_in_range))
    
    # Print all results, each on a new line, followed by a final newline.
    sys.stdout.write("
".join(results) + "
")

solve()