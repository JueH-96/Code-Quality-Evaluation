def main():
    import sys, bisect
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # There are (N-1)//2 sleep sessions.
    num_intervals = (N - 1) // 2
    sleep_starts = []
    sleep_ends = []
    # The sleep intervals are defined as:
    # For each i with 1 ≤ i ≤ (N-1)/2, sleep session i is from A[2*i] to A[2*i+1] (1-indexed).
    # In our 0-indexed list A, that becomes: interval from A[2*i-1] to A[2*i].
    for i in range(num_intervals):
        s = A[2 * i + 1]  # sleep start time
        e = A[2 * i + 2]  # wake-up time
        sleep_starts.append(s)
        sleep_ends.append(e)
    
    # Precompute prefix sum of sleep durations over the sessions.
    prefix = [0] * (num_intervals + 1)
    for i in range(num_intervals):
        prefix[i + 1] = prefix[i] + (sleep_ends[i] - sleep_starts[i])
    
    # Define a function that computes the total sleep time from time 0 to time x.
    def sleep_time(x):
        # Find the number of sleep intervals that ended by time x (using bisect_right,
        # which counts an interval as finished if x equals its end time).
        i = bisect.bisect_right(sleep_ends, x)
        total = prefix[i]
        # Check if the next sleep interval (if any) is partially in progress at time x.
        if i < num_intervals and sleep_starts[i] < x:
            # x lies within the interval starting at sleep_starts[i]. Add the partial sleep time.
            total += x - sleep_starts[i]
        return total

    Q = int(next(it))
    out_lines = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        # The answer for a query [l, r] is the sleep time from 0 to r minus sleep time from 0 to l.
        ans = sleep_time(r) - sleep_time(l)
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()