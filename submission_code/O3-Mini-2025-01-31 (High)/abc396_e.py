def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return

    # DSU with XOR potential:
    parent = list(range(n+1))
    xor_val = [0]*(n+1)  # xor_val[i] stores the XOR-weight from node i to its parent

    def find(x):
        if parent[x] != x:
            root, pot = find(parent[x])
            xor_val[x] ^= pot
            parent[x] = root
            return parent[x], xor_val[x]
        return x, xor_val[x]

    def union(u, v, d):
        ru, pu = find(u)
        rv, pv = find(v)
        if ru == rv:
            # They are in the same component; check consistency.
            if (pu ^ pv) != d:
                return False
            return True
        # Merge ru into rv. We need to set the new xor_val for ru, namely:
        # xor_val[ru] = pu ^ pv ^ d  so that for any node x in ru: A[x] = (B ⊕ r[x])
        parent[ru] = rv
        xor_val[ru] = pu ^ pv ^ d
        return True

    possible = True
    for _ in range(m):
        try:
            u = int(next(it))
            v = int(next(it))
            d = int(next(it))
        except StopIteration:
            break
        # If an edge has the same vertex twice, then we must have 0 = d.
        if u == v:
            if d != 0:
                possible = False
                break
            continue
        if not union(u, v, d):
            possible = False
            break
    if not possible:
        sys.stdout.write("-1")
        return

    # For every node, compute its representative and the potential (r[i]) relative to that rep.
    comp = {}      # rep -> list of nodes in that component.
    rep_of = [0]*(n+1)      # rep_of[i] stores the representative of node i.
    potential = [0]*(n+1)   # potential[i] = computed r[i]
    for i in range(1, n+1):
        rep, pot = find(i)
        rep_of[i] = rep
        potential[i] = pot
        if rep not in comp:
            comp[rep] = []
        comp[rep].append(i)
    
    # For every component we have freedom to choose B (the base)
    # so that A[i] = B ⊕ (computed potential) for i in that component.
    # Choose B bitwise to minimize the sum of A[i].
    BIT_COUNT = 31  # This is enough since Z_i <= 10^9 so numbers < 2^31.
    base_for = {}   # rep -> best chosen free parameter B.
    for rep, nodes in comp.items():
        vals = [ potential[i] for i in nodes ]
        B = 0
        for k in range(BIT_COUNT):
            cnt0 = 0
            for val in vals:
                if ((val >> k) & 1) == 0:
                    cnt0 += 1
            cnt1 = len(vals) - cnt0
            # If B_k = 0, then each vertex with r[i]'s kth bit = 1 adds 2^k.
            # If B_k = 1, then each vertex with r[i]'s kth bit = 0 adds 2^k.
            if cnt0 < cnt1:
                B |= (1 << k)
        base_for[rep] = B

    # Now compute answer array: for each vertex i, A[i] = base_for[rep_of[i]] ⊕ potential[i].
    ans = [0]*(n+1)
    for i in range(1, n+1):
        ans[i] = base_for[ rep_of[i] ] ^ potential[i]
    
    sys.stdout.write(" ".join(str(x) for x in ans[1:]))

if __name__ == '__main__':
    main()