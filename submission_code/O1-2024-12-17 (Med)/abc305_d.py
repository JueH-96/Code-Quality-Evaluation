def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Read N
    N = int(input_data[0])
    # Read array A
    A = list(map(int, input_data[1:N+1]))
    # Number of queries
    Q = int(input_data[N+1])
    # Queries start from input_data[N+2]
    queries = input_data[N+2:]
    
    # We have an odd-length array A, with A[0] = 0 and intervals of sleep are
    # [A[1], A[2]], [A[3], A[4]], ..., [A[N-2], A[N-1]] if we view A in 0-based indexing.
    #
    # Let M = (N - 1) // 2 be the number of such sleep intervals.
    # For i in [0..M-1], the i-th sleep interval is [A[2i+1], A[2i+2]].
    #
    # We'll build arrays S and E of length M:
    #   S[i] = A[2i+1] (the start of the i-th sleep interval)
    #   E[i] = A[2i+2] (the end   of the i-th sleep interval)
    #
    # Then we build a prefix-sum array C of length M+1 where:
    #   C[0] = 0
    #   C[i+1] = C[i] + (E[i] - S[i])   for i in [0..M-1]
    # Thus C[i] is the total of all completed sleep intervals up to (but not including) interval i.
    #
    # To compute total sleep from time 0 up to time x, do:
    #   1) Use bisect_right(E, x) -> idx. Then i = idx - 1.
    #      This i is the last interval that ends at or before x.
    #   2) ans = 0
    #      If i >= 0, ans += C[i+1]  (sum of all fully completed intervals up to i)
    #   3) Check partial overlap for interval i+1 (which is the next interval),
    #      if i+1 < M:
    #          start = S[i+1]
    #          end   = E[i+1]
    #          if x > start:
    #             ans += min(x, end) - start
    #   4) Return ans
    #
    # Finally, for each query [l, r], the answer is totalSleep(r) - totalSleep(l).

    M = (N - 1) // 2
    
    S = [0] * M  # starts
    E = [0] * M  # ends
    for i in range(M):
        S[i] = A[2*i + 1]
        E[i] = A[2*i + 2]
    
    # Build prefix sums of interval lengths
    C = [0] * (M + 1)
    for i in range(M):
        C[i+1] = C[i] + (E[i] - S[i])
    
    def total_sleep(x):
        # If x < S[0], no interval is fully completed, might be partial in 1st interval
        # Use bisect_right to find how many intervals end <= x
        idx = bisect.bisect_right(E, x)  # number of intervals that end on or before x
        i = idx - 1  # the last interval fully contained in [0..x]
        
        ans = 0
        if i >= 0:
            ans += C[i+1]  # all fully completed intervals up to i
        
        # check partial overlap of the next interval i+1
        if i+1 < M:
            st = S[i+1]
            if x > st:
                ans += max(0, min(x, E[i+1]) - st)
        
        return ans
    
    # Process queries
    pos = 0
    out = []
    for _ in range(Q):
        l = int(queries[pos]); r = int(queries[pos+1])
        pos += 2
        # total minutes asleep in [l, r] = total_sleep(r) - total_sleep(l)
        out.append(str(total_sleep(r) - total_sleep(l)))
    
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()