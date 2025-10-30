def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:N+1]))
    Q = int(input_data[N+1])
    LR = input_data[N+2:]

    # Build prefix array P where P[i] = total sleep from time 0 up to A[i].
    # Recall that Takahashi sleeps in intervals [A[2k+1], A[2k+2]] (1-based),
    # which in 0-based indexing becomes [A[1],A[2]], [A[3],A[4]], ...
    # So for i >= 1, if i is even, P[i] = P[i-1] + (A[i] - A[i-1]) else P[i] = P[i-1].
    P = [0]*N
    for i in range(1, N):
        P[i] = P[i-1]
        if i % 2 == 1:  # i is odd => A[i-1] to A[i] is a sleeping interval in 0-based
            P[i] += A[i] - A[i-1]

    # Function to get the total sleep from time 0 up to (and including) t
    def sleep_amount(t):
        # Find the largest j where A[j] <= t
        # j = bisect_right(A, t) - 1
        j = bisect.bisect_right(A, t) - 1
        if j < 0:
            return 0
        ans = P[j]
        # If j is odd, that means [A[j], A[j+1]] is a sleeping interval,
        # so add partial from A[j] up to t
        if j % 2 == 1:
            ans += (t - A[j])
        return ans

    # Process each query
    idx = 0
    out = []
    for _ in range(Q):
        l = int(LR[idx]); r = int(LR[idx+1])
        idx += 2
        out.append(str(sleep_amount(r) - sleep_amount(l)))

    print("
".join(out))

# Do not forget to call main
if __name__ == "__main__":
    main()