def main():
    import sys
    from itertools import combinations

    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    K = int(next(it))

    uu = [0] * M
    vv = [0] * M
    ww = [0] * M
    for i in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        uu[i] = u
        vv[i] = v
        ww[i] = w

    # We'll brute‐force all subsets of edges of size N-1 and check if they form a spanning tree.
    # Track the minimum sum of weights mod K.
    ans = K  # larger than any possible mod-K sum

    # Pre-allocate a parent array placeholder
    # (we'll reinitialize per combination)
    for comb in combinations(range(M), N - 1):
        # Union-Find initialization
        parent = list(range(N))
        comp = N
        total_w = 0
        valid = True

        for ei in comb:
            # find root of uu[ei]
            x = uu[ei]
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            ru = x
            # find root of vv[ei]
            x = vv[ei]
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            rv = x
            # if they are already connected, it's a cycle
            if ru == rv:
                valid = False
                break
            # otherwise union
            parent[ru] = rv
            comp -= 1
            total_w += ww[ei]

        if not valid or comp != 1:
            continue

        m = total_w % K
        if m < ans:
            ans = m
            if ans == 0:
                # can't get better than 0
                break

    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()