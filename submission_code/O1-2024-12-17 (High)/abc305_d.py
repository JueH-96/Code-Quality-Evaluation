def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    idx = 0

    # Read N and the log array B
    N = int(data[idx]); idx += 1
    B = list(map(int, data[idx:idx+N]))
    idx += N

    # Precompute S[i] = total minutes asleep from time 0 up to B[i].
    # Sleep intervals in 0-based indexing are [B[1],B[2]], [B[3],B[4]], ...
    # so at every even i (2,4,6,...) we add (B[i] - B[i-1]).
    S = [0]*N
    for i in range(1, N):
        S[i] = S[i-1]
        if i % 2 == 0:  # i is even => this completes a sleep interval
            S[i] += B[i] - B[i-1]

    # A helper function to get total slept minutes from t=0 up to time x
    def total_sleep(x):
        if x <= B[0]:
            return 0
        if x >= B[-1]:
            return S[-1]
        k = bisect.bisect_right(B, x) - 1  # largest index where B[k] <= x
        # If k is odd, we are inside a sleeping interval [B[k], B[k+1]]
        if k % 2 == 1:
            return S[k] + (x - B[k])
        else:
            return S[k]

    # Read Q and process each query
    Q = int(data[idx]); idx += 1
    out = []
    for _ in range(Q):
        l = int(data[idx]); idx += 1
        r = int(data[idx]); idx += 1
        out.append(str(total_sleep(r) - total_sleep(l)))

    print("
".join(out))

# Call main at the end
if __name__ == "__main__":
    main()