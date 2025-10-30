def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    S = data[2]
    queries = data[3:]

    # Precompute an array A where A[i] = 1 if S[i] == S[i+1], else 0 (1-based indexing)
    A = [0] * N
    for i in range(N - 1):
        A[i + 1] = 1 if S[i] == S[i + 1] else 0

    # Compute prefix sums of A
    prefixSums = [0] * (N + 1)
    for i in range(1, N):
        prefixSums[i] = prefixSums[i - 1] + A[i]

    # Process each query
    idx = 0
    out = []
    for _ in range(Q):
        l = int(queries[idx]); r = int(queries[idx + 1])
        idx += 2
        # Sum from A[l] to A[r-1] = prefixSums[r-1] - prefixSums[l-1]
        out.append(str(prefixSums[r - 1] - prefixSums[l - 1]))

    print("
".join(out))

# Call solve
solve()