import sys
from bisect import bisect_right

# YOUR CODE HERE
def solve():
    """
    Solves the Takahashi's Sleep Log problem by using a cumulative sum approach.
    """

    # Set fast I/O
    input = sys.stdin.readline

    # Read problem inputs
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, input().split()))
        Q_str = input()
        if not Q_str: return
        Q = int(Q_str)
    except (IOError, ValueError):
        return

    # --- Precomputation Step ---
    # The core idea is to create a function S(t) that calculates the total sleep time
    # from time 0 to t. The answer for a query (l, r) is then S(r) - S(l).
    #
    # To implement S(t) efficiently, we precompute the cumulative sleep durations.
    # The sleep intervals are [A_2, A_3], [A_4, A_5], ... (using 1-based problem indexing).
    # In 0-based list indexing, these correspond to [A[1], A[2]], [A[3], A[4]], etc.

    num_sleep_intervals = (N - 1) // 2
    # prefix_sleep[i] will store the sum of durations of the first i sleep intervals.
    # Size is num_sleep_intervals + 1 for easier 1-based indexing of intervals.
    prefix_sleep = [0] * (num_sleep_intervals + 1)

    for i in range(num_sleep_intervals):
        # The i-th sleep interval (0-indexed) is from A[2*i + 1] to A[2*i + 2].
        sleep_start = A[2 * i + 1]
        sleep_end = A[2 * i + 2]
        duration = sleep_end - sleep_start
        prefix_sleep[i + 1] = prefix_sleep[i] + duration

    # --- S(t) Function Definition ---
    def get_total_sleep_up_to(t):
        """
        Calculates the total sleep time in the interval [0, t].
        
        Args:
            t: The time up to which to calculate total sleep.
        
        Returns:
            The total minutes asleep from time 0 up to time t.
        """
        # Use binary search to find which interval of events t falls into.
        # bisect_right(A, t) returns an insertion point k. This means that for
        # index idx = k - 1, we have A[idx] <= t. If k < N, then t < A[k].
        k = bisect_right(A, t)
        idx = k - 1
        
        # If t is before the first event (A[0]=0), total sleep is 0.
        # This also handles t<0, though problem constraints ensure t>=0.
        if idx < 0:
            return 0
        
        # The state (awake/asleep) depends on the index's parity.
        # Even indices (0, 2, 4...) in the 0-based A list mark the start of an awake period.
        # Odd indices (1, 3, 5...) mark the start of a sleep period.
        
        # At time A[idx], the number of full sleep intervals that have passed is idx // 2.
        num_full_sleeps = idx // 2
        total_sleep = prefix_sleep[num_full_sleeps]
        
        # If t falls within a sleep period (i.e., the period starts at an odd index),
        # we need to add the partial duration from the start of this period up to t.
        if idx % 2 == 1:
            # The current sleep period started at A[idx].
            total_sleep += t - A[idx]
            
        return total_sleep

    # --- Process Queries ---
    # We process each query by calculating S(r) - S(l).
    # The results are collected in a list and printed all at once for efficiency.
    output_lines = []
    for _ in range(Q):
        line = input()
        if not line: break
        l, r = map(int, line.split())
        ans = get_total_sleep_up_to(r) - get_total_sleep_up_to(l)
        output_lines.append(str(ans))

    # Print all results, separated by newlines.
    print("
".join(output_lines))

solve()