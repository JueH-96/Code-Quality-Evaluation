def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    X = [int(next(it)) for _ in range(M)]
    
    # Explanation:
    # We have islands 1,...,N in a cycle with N bridges.
    # When we remove one bridge (say the one connecting island r and r+1 with r from 1 to N,
    # where r = N means the bridge between island N and 1), the remaining graph becomes a tree with
    # a forced linear (chain) ordering of islands. In particular, if we remove the r-th bridge,
    # the islands in the chain appear in the order:
    #   T_r = [r+1, r+2, …, N, 1, 2, …, r].
    #
    # For a tour whose required visit order is X_1, X_2, …, X_M,
    # the minimal number of bridge crossings is the sum (over consecutive required islands)
    # of the distances in this tree.
    # Because the tree is a line, the distance between two islands is the absolute difference of
    # their positions in the T_r ordering.
    # It turns out that for two islands u and v (with u != v), if we let a = min(u,v) and b = max(u,v),
    # then:
    # • If both u and v lie in the same segment of T_r (either both > r or both <= r),
    #   the distance is |u - v|.
    # • Otherwise (i.e. if one is in the segment {r+1,...,N} and the other in {1,...,r}),
    #   the distance is N - |u - v|.
    #
    # Notice that this is exactly like choosing between the shorter circular arc and the longer one.
    #
    # Rewriting the above in a uniform way:
    # For each consecutive pair (X_i, X_{i+1}), let d = |X_i - X_{i+1]|.
    # Then if r (the bridge removed, where r is in {1,...,N}) is chosen,
    # the distance for that segment is:
    #    d                    if r is not in [min(X_i, X_{i+1]), max(X_i, X_{i+1]) - 1],
    #    (N - d)              if r is in [min(X_i, X_{i+1]), max(X_i, X_{i+1]) - 1].
    # This can be written as:
    #    d + I[r in [a, b-1]] * ((N - d) - d)
    # =  d + I[r in [a, b-1]] * (N - 2*d),
    # where a = min(X_i, X_{i+1]) and b = max(X_i, X_{i+1]).
    #
    # Let S0 be the sum of base distances for all consecutive visited islands:
    #   S0 = sum(|X_{i+1} - X_i|)  for i = 1 to M-1.
    # Then the total tour length when bridge r is removed is:
    #   F(r) = S0 + sum_{each pair with a = min, b = max} [I[r in [a, b-1]] * (N - 2*(b-a))].
    #
    # To compute the contribution over all pairs as r varies, we use a difference array.
    # For each consecutive pair with a and b and diff = b-a, if we define
    #   additional = (N - 2*diff)
    # then for all r in the range a to b-1 we add that additional difference.
    #
    # Using a difference array diff_arr (of length N+2 for safety),
    # we do for each transition:
    #   diff_arr[a] += (N - 2*(b-a))
    #   diff_arr[b] -= (N - 2*(b-a))
    #
    # Afterwards, for each r from 1 to N, the additional value is the prefix sum up to index r.
    # And the final answer is min_{r=1 to N} (S0 + prefix[r]).
    
    S0 = 0
    diff_arr = [0] * (N + 2)
    
    for i in range(M - 1):
        u = X[i]
        v = X[i + 1]
        if u < v:
            a = u
            b = v
        else:
            a = v
            b = u
        d = b - a
        S0 += d
        add_val = N - 2 * d  # additional contribution if r is in [a, b-1]
        diff_arr[a] += add_val
        diff_arr[b] -= add_val

    best = None
    prefix = 0
    for r in range(1, N + 1):
        prefix += diff_arr[r]
        tour_length = S0 + prefix
        if best is None or tour_length < best:
            best = tour_length
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()