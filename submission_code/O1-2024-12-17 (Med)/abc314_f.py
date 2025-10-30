def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    p = [0]*(N-1)
    q = [0]*(N-1)
    idx = 1
    for i in range(N-1):
        p[i] = int(input_data[idx]); q[i] = int(input_data[idx+1])
        idx += 2

    MOD = 998244353

    # -------------------------------------------------------------------------
    # Precompute inverse of all integers up to N*2 (sA + sB can be at most 2*N, 
    # but strictly at most 2*N because we unify step by step). 
    # However, up to N is actually sufficient because sA+sB <= N for each match.
    # We'll do up to (N) or (2*N) safely.  Here we do up to N (≤ 2×10^5).
    # -------------------------------------------------------------------------
    max_val = N  # sA + sB cannot exceed N at any step.
    inv = [0]*(max_val+1)
    inv[1] = 1
    for i in range(2, max_val+1):
        # Fermat's little theorem method in O(1):
        inv[i] = MOD - (MOD//i) * inv[MOD % i] % MOD

    # -------------------------------------------------------------------------
    # Union-Find with "offset" to track expected-win offsets 
    # and "baseWins" for each root cluster.
    # -------------------------------------------------------------------------
    parent = list(range(N+1))
    size = [1]*(N+1)
    baseWins = [0]*(N+1)  # baseWins[r] = expected wins for each node in r's cluster (root r).
    offset = [0]*(N+1)    # offset[x] so that E[x] = baseWins[find(x)] + offset[x].

    def find(x):
        if parent[x] != x:
            px = find(parent[x])
            # Add parent's offset to mine
            offset[x] = (offset[x] + offset[parent[x]]) % MOD
            parent[x] = px
        return parent[x]

    def union(a, b):
        # Attach b's root to a's root
        parent[b] = a
        # We need offset[b] = baseWins[a] - baseWins[b] (mod M),
        # so that for any x in cluster b, E[x] can be re-routed consistently.
        offset[b] = (baseWins[a] - baseWins[b]) % MOD
        size[a] += size[b]

    # -------------------------------------------------------------------------
    # Process each match
    # -------------------------------------------------------------------------
    for i in range(N-1):
        rp = find(p[i])
        rq = find(q[i])
        sp = size[rp]
        sq = size[rq]
        # Probability that rp's team (first) wins = sp / (sp + sq)
        # Probability that rq's team (second) wins = sq / (sp + sq)
        # We do everything mod M.  (sp + sq) <= N, so it's safe to invert.
        denom_inv = inv[sp + sq]  # inverse of sp+sq
        fracA = (sp * denom_inv) % MOD  # sp/(sp+sq)
        fracB = (sq * denom_inv) % MOD  # sq/(sp+sq)

        # Add those to the clusters' baseWins
        baseWins[rp] = (baseWins[rp] + fracA) % MOD
        baseWins[rq] = (baseWins[rq] + fracB) % MOD

        # Union by size to keep tree shallow:
        # Attach the smaller root to the larger root
        if size[rp] < size[rq]:
            rp, rq = rq, rp
        # now rp is bigger
        union(rp, rq)

    # -------------------------------------------------------------------------
    # Final expected wins for each player i:
    # E[i] = baseWins[ find(i) ] + offset[i]  (mod M)
    # -------------------------------------------------------------------------
    # Do one more pass of find to ensure path compression and final offset
    for i in range(1, N+1):
        find(i)
    # Now compute and print
    ans = [(baseWins[parent[i]] + offset[i]) % MOD for i in range(1, N+1)]
    print(" ".join(map(str, ans)))


# Call main() to ensure the solution is executed.
if __name__ == "__main__":
    main()