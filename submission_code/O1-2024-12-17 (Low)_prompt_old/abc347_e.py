def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    xs = list(map(int, input_data[2:]))

    # Track whether x is currently in set S, and the query index it was last toggled on
    inS = [False] * (N+1)
    last_on = [-1] * (N+1)

    # For each x, we will store intervals [L, R] meaning x was in S from query L through query R
    intervals = [[] for _ in range(N+1)]

    # We'll track the size of S after each query in sizeS[i]
    sizeS = [0] * (Q+1)
    current_size = 0  # current number of elements in S

    # Process queries
    for i in range(1, Q+1):
        x = xs[i-1]
        if not inS[x]:
            # x goes from out-of-S to in-S
            inS[x] = True
            last_on[x] = i
            current_size += 1
        else:
            # x goes from in-S to out-of-S
            # x was in S for queries [last_on[x], i-1]
            intervals[x].append((last_on[x], i-1))
            inS[x] = False
            current_size -= 1
        sizeS[i] = current_size

    # For those still in S at the end, they remained in from last_on[x] through Q
    for x in range(1, N+1):
        if inS[x]:
            intervals[x].append((last_on[x], Q))

    # Build prefix sum of sizeS to quickly compute sum of |S| over intervals
    pS = [0] * (Q+1)
    for i in range(1, Q+1):
        pS[i] = pS[i-1] + sizeS[i]

    # Compute final A
    A = [0] * N
    for x in range(1, N+1):
        for (start_q, end_q) in intervals[x]:
            if start_q <= end_q:
                A[x-1] += pS[end_q] - pS[start_q - 1]

    print(" ".join(map(str, A)))

# Uncomment the following to run locally (but keep it commented on submission):
# if __name__ == "__main__":
#     solve()