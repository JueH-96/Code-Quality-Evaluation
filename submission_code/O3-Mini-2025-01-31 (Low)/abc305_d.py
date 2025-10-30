def main():
    import sys
    import bisect
    input = sys.stdin.readline

    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    Q = int(input())
    
    # Build sleep intervals: each interval is (start, end)
    # Sleep sessions: [A[1], A[2]] in 0-indexing: Actually given session for i in 1...((N-1)//2):
    #    session i: falls asleep at A[2*i] and wakes up at A[2*i+1]   (since A[0] = 0, A[1] is first wake time, etc.)
    # For i from 1 to (N-1)//2, in 0-index: start = A[2*i], end = A[2*i+1]
    sleep_intervals = []
    # Total number of sessions = (N-1)//2
    sessions = (N - 1) // 2
    for i in range(1, sessions + 1):
        start = A[2*i]
        end = A[2*i+1]
        sleep_intervals.append((start, end))
    
    # Precompute arrays of starts and ends for binary search and cumulative sleep durations.
    starts = []
    ends = []
    durations = []
    for s, e in sleep_intervals:
        starts.append(s)
        ends.append(e)
        durations.append(e - s)
        
    # prefix sum of durations
    prefix = [0]
    for d in durations:
        prefix.append(prefix[-1] + d)
    # Now prefix[i] = total sleep duration for intervals 0 to i-1.
    
    # Process queries
    results = []
    for _ in range(Q):
        l, r = map(int, input().split())
        total_sleep = 0
        
        # Find the first interval which may have end > l
        idx_left = bisect.bisect_right(ends, l-1)  # first interval where end >= l
        # Find the last interval which may have start < r
        idx_right = bisect.bisect_left(starts, r) - 1  # last interval with start < r
        
        # No interval overlapping query range
        if idx_left > idx_right or idx_left < 0 or idx_right < 0 or idx_left >= len(sleep_intervals):
            results.append(0)
            continue
        
        # idx_left and idx_right are indices in sleep_intervals that might contribute.
        # Compute total sleep by handling left partial, full intervals in between, and right partial.
        # For the leftmost interval (at idx_left):
        s, e = sleep_intervals[idx_left]
        left_contrib = 0
        if e <= l:
            left_contrib = 0
        else:
            left_contrib = min(e, r) - max(s, l)
            if left_contrib < 0:
                left_contrib = 0
        
        if idx_left == idx_right:
            total_sleep = left_contrib
        else:
            # Rightmost interval contribution (at idx_right)
            s_r, e_r = sleep_intervals[idx_right]
            right_contrib = min(e_r, r) - max(s_r, l)
            if right_contrib < 0:
                right_contrib = 0
            # Full intervals between the leftmost and rightmost:
            full = 0
            if idx_right > idx_left + 1:
                full = prefix[idx_right] - prefix[idx_left+1]
            total_sleep = left_contrib + full + right_contrib
            
        results.append(total_sleep)
    
    # Output results
    print("
".join(map(str, results)))
    
if __name__ == '__main__':
    main()