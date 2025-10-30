def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    queries = [int(next(it)) for _ in range(Q)]
    
    # We'll track membership (indices 1..N) and record the query numbers where an index is toggled.
    membership = [False] * (N + 1)
    toggles = [[] for _ in range(N + 1)]
    
    # v[q] will store the size |S| after processing the q-th query (queries are 1-indexed).
    v = [0] * (Q + 1)
    current_size = 0

    # Process each query in order.
    for q in range(1, Q + 1):
        x = queries[q - 1]
        if not membership[x]:
            membership[x] = True
            current_size += 1
        else:
            membership[x] = False
            current_size -= 1
        toggles[x].append(q)  # record that x toggled at query number q
        v[q] = current_size

    # Build a prefix sum array P where P[i] = v[1] + v[2] + ... + v[i] for i = 1,...,Q (with P[0] = 0).
    P = [0] * (Q + 1)
    for i in range(1, Q + 1):
        P[i] = P[i - 1] + v[i]
    
    # At the end, each element A[k] (for k from 1 to N) has accumulated contributions:
    # A[k] = sum_{q : k in S after q} |S| = sum_{q in intervals where k is active} v[q].
    # For each index k, toggles[k] is a sorted list of query numbers at which k was toggled.
    # Since all indices start off:
    #   - After the 1st toggle, k becomes active until the next toggle.
    #   - After the 2nd toggle, it becomes inactive, and so on.
    # Thus if toggles[k] = [t1, t2, ..., tm],
    # the index k is active on queries t1..(t2-1), t3..(t4-1), ... and if m is odd then also t_m..Q.
    # We can sum the contributions over these intervals with the prefix sum array P.
    
    # Compute A's for indices 1..N
    ans = [0] * (N + 1)
    for k in range(1, N + 1):
        if not toggles[k]:
            continue
        total = 0
        arr = toggles[k]
        m = len(arr)
        for j in range(0, m, 2):
            L = arr[j]                   # start of an active interval
            if j + 1 < m:
                R = arr[j + 1] - 1       # active until the query before the next toggle
            else:
                R = Q                  # if count is odd, continue until the end
            if L <= R:
                total += P[R] - P[L - 1]
        ans[k] = total

    # Format and write the output. We output A[1] through A[N].
    sys.stdout.write(" ".join(str(ans[k]) for k in range(1, N + 1)))


if __name__ == '__main__':
    main()