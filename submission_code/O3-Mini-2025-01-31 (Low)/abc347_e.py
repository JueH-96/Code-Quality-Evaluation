def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    Q = int(next(it))
    queries = [int(next(it)) for _ in range(Q)]
    
    # We'll simulate the queries to compute f[i]: the size of S after the i-th query.
    # Also, for each index j (1-indexed) we record the queries when j is toggled.
    f = [0] * (Q + 1)  # Using 1-indexing for queries
    toggles = [[] for _ in range(N + 1)]
    state = [False] * (N + 1)  # Current membership status in S (False means not presently in S)
    current_size = 0
    
    for i, x in enumerate(queries, start=1):
        if state[x]:
            current_size -= 1
            state[x] = False
        else:
            current_size += 1
            state[x] = True
        f[i] = current_size
        toggles[x].append(i)
    
    # Precompute prefix sums of f for quick range sum queries.
    # pf[i] = f[1] + f[2] + ... + f[i]
    pf = [0] * (Q + 1)
    for i in range(1, Q + 1):
        pf[i] = pf[i - 1] + f[i]
    
    # The value of A[j] is the sum over all queries in which j was in S.
    # For each j, we know the queries where x==j was toggled.
    # Initially, j is not in S. When it is toggled (at query time t), it gets added if it wasn’t present.
    # Hence for j, if toggling occurs at times t1, t2, t3, t4, ... then j is active during intervals:
    # [t1, t2-1], [t3, t4-1], ... and if the count of toggles is odd then j remains active up to Q after the last toggle.
    # The sum of f values over a query interval [l, r] is pf[r]-pf[l-1].
    res = [0] * (N + 1)
    for j in range(1, N + 1):
        if not toggles[j]:
            continue
        times = toggles[j]
        # Iterate in pairs: the j-th value is active from times[idx] (insertion) until one before times[idx+1] (removal)
        for idx in range(0, len(times), 2):
            l = times[idx]
            # If there is a next toggle, it is removal so the active interval is [l, times[idx+1]-1]
            # Otherwise, j remains active until the end (Q)
            r = times[idx + 1] - 1 if idx + 1 < len(times) else Q
            if l <= r:
                res[j] += pf[r] - pf[l - 1]
    
    # Output the final sequence A_1, A_2, ..., A_N
    sys.stdout.write(" ".join(str(res[j]) for j in range(1, N + 1)))

if __name__ == '__main__':
    main()