def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    Q = int(next(it))
    
    # We interpret the log as follows (0-indexed):
    # A[0] is when Takahashi got up and is awake.
    # For each sleep session, he goes to bed at A[i] for odd i (i=1,3,5,...) 
    # and wakes up at A[i+1] for even i (i+1=2,4,6,...).
    # So the sleep intervals are (A[1], A[2]), (A[3], A[4]), etc.
    #
    # To answer queries asking the total sleep time between times l and r,
    # we define a function f(x) which gives the total sleep time in [0, x].
    #
    # To do this efficiently, we precompute a prefix array P such that
    # P[i] represents the total sleep time accumulated up to time A[i].
    # For indices:
    #   - i = 0: A[0] (wake-up time, no sleep completed)
    #   - for an odd i, A[i] is a sleep start so no sleep is added at that moment.
    #   - for an even i (i >= 2), A[i] is a wake-up time; the sleep session
    #     from A[i-1] (the sleep start) to A[i] is complete, so we add (A[i]-A[i-1]).
    #
    # With this prefix array we can later compute f(x) using binary search on A.
    
    P = [0] * N
    for i in range(1, N):
        if i % 2 == 0:
            P[i] = P[i - 1] + (A[i] - A[i - 1])
        else:
            P[i] = P[i - 1]
    
    # f(x) returns the total sleep time in [0, x].
    # We use bisect_right to find how many events in A have occurred by time x.
    # Two cases:
    #   1. If x falls inside a sleep session: the last event before x was when he went to bed.
    #      In that case, j is even and x is in the middle of a sleep session that started at A[j-1],
    #      so f(x) = P[j-1] + (x - A[j-1]).
    #   2. If x falls in a wake period: then all past sleep sessions are complete so
    #      f(x) = P[j-1] (where j is odd) because the sleep session in progress is not active.
    def f(x):
        j = bisect.bisect_right(A, x)
        if j == 0:
            return 0
        if j % 2 == 0:
            return P[j - 1] + (x - A[j - 1])
        else:
            return P[j - 1]
    
    # For each query [l, r], the sleep time in [l, r] is f(r) - f(l).
    output_lines = []
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        sleep_time = f(r) - f(l)
        output_lines.append(str(sleep_time))
    
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()