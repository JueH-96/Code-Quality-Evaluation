def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    x = list(map(int, data[2:]))

    # T[i] will store the size of set S AFTER processing the i-th query (1-based indexing for convenience)
    T = [0]*(Q+1)

    # Keep track of current membership (whether an index is in S) and current size of S
    in_set = [False]*(N+1)
    size_S = 0
    
    # We also need to record when each j toggles (enters or leaves S)
    toggles = [[] for _ in range(N+1)]

    # Process queries to fill T and record toggle positions
    for i in range(1, Q+1):
        val = x[i-1]
        if in_set[val]:
            # val is currently in S, so remove it
            in_set[val] = False
            size_S -= 1
        else:
            # val is not in S, so add it
            in_set[val] = True
            size_S += 1
        T[i] = size_S
        
        # Record that index val toggled at query i
        toggles[val].append(i)

    # Build prefix sums of T to quickly sum over intervals
    # P[i] = T[1] + T[2] + ... + T[i]
    P = [0]*(Q+1)
    for i in range(1, Q+1):
        P[i] = P[i-1] + T[i]

    # Now compute result for each j in [1..N] using its toggle intervals
    A = [0]*N
    for j in range(1, N+1):
        arr = toggles[j]
        if not arr:
            continue
        # arr gives the 1-based indices of queries where j toggles
        # The set is initially empty, so toggles alternate in-out-in-out...
        # The intervals where j is "in" are [arr[2k], arr[2k+1]) for each valid pair
        # If there's an odd number of toggles, the final interval goes up to Q+1
        m = len(arr)
        for k in range(0, m, 2):
            start = arr[k]
            if k+1 < m:
                end = arr[k+1]  # j goes out at end
            else:
                end = Q+1       # stays in until after all queries

            # sum of T[i] for i in [start..end-1]
            # = P[end-1] - P[start-1], because P[i] = sum of T up to i
            A[j-1] += (P[end-1] - P[start-1])

    # Print the final sequence A
    print(" ".join(map(str, A)))


# Don't forget to call main()
if __name__ == "__main__":
    main()