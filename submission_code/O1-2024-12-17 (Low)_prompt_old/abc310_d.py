def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, T, M = map(int, input_data[:3])
    edges = input_data[3:]
    
    # Adjacency matrix for N up to 10
    # We'll mark adjacency[i][j] = True if i and j are incompatible
    adj = [[False]*N for _ in range(N)]
    idx = 0
    for _ in range(M):
        a = int(edges[idx]) - 1
        b = int(edges[idx+1]) - 1
        idx += 2
        adj[a][b] = adj[b][a] = True

    # Precompute "independent[mask]" = True if the subset of vertices in 'mask' is an independent set
    # i.e. no two of them are incompatible.
    independent = [True]*(1<<N)
    for mask in range(1, 1<<N):
        # a quick check can be done by comparing pairs
        # but since N=10, it's not too large.
        # We'll break early if we find an incompatible pair.
        # Extract elements of mask
        # possible check: for i < j in mask, if adj[i][j], then independent[mask] = False
        # We'll do a simple double loop up to N=10.
        bits = []
        # collect the indices in mask
        s = mask
        v = 0
        while s:
            if s & 1:
                bits.append(v)
            v += 1
            s >>= 1
        # check pairs
        ok = True
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                if adj[bits[i]][bits[j]]:
                    ok = False
                    break
            if not ok:
                break
        independent[mask] = ok

    # We want F(x) = the number of ways to color the full set of vertices
    # with x labeled colors (not necessarily using all x).
    # We'll use a DP approach:
    # dp[S][c] = number of ways to color the subset S with c (labeled) colors
    # Recurrence:
    #   dp[S][c] = sum_{I subset of S, I is independent} dp[S\I][c-1]
    # with base cases:
    #   dp[0][0] = 1, dp[non-empty][0] = 0
    # we only need c up to T, and S up to (1<<N)-1.
    
    dp = [[0]*(T+1) for _ in range(1<<N)]
    # dp for empty set
    dp[0][0] = 1  # 1 way to color empty set with 0 colors
    
    from itertools import combinations

    # We'll iterate subsets in increasing order of size
    # because dp[S][c] depends on dp[S\I][c-1], whose size is smaller
    # We'll build a list of subsets by ascending cardinality
    subsets_by_size = [[] for _ in range(N+1)]
    for s in range(1<<N):
        subsets_by_size[bin(s).count("1")].append(s)

    for size_s in range(N+1):
        for S in subsets_by_size[size_s]:
            for c in range(1, T+1):
                # do the transition dp[S][c] = sum of dp[S^I][c-1] for all I subset of S that is independent
                # We'll iterate over submasks of S
                # but enumerating large subsets can be 2^size_s up to 2^N=1024 in the worst case, which is still okay.
                ans = 0
                # submask enumeration trick: let sub = S
                # sub = (sub-1) & S to get the next submask
                sub = S
                while True:
                    # check if sub is independent
                    if independent[sub]:
                        remainder = S ^ sub  # S \ sub
                        ans += dp[remainder][c-1]
                    if sub == 0:
                        break
                    sub = (sub - 1) & S
                dp[S][c] = ans

    # F(c) = dp[(1<<N)-1][c]
    # Then the answer = sum_{k=0..T} ((-1)^k * C(T,k) * F(T-k)) by inclusion-exclusion
    # to ensure exactly T distinct colors are used.
    
    # We'll build a small nCr table for up to n=10
    from math import comb
    
    full_mask = (1<<N) - 1
    
    def F(x):
        return dp[full_mask][x]
    
    ans = 0
    sign = 1
    # sum_{k=0..T} (-1)^k * C(T,k) * F(T-k)
    # Because we want EXACTLY T distinct colors used.
    for k in range(T+1):
        # T-k -> color count
        val = F(T-k)
        # combine with combinatorial factor
        cval = comb(T, k)
        if k % 2 == 1:
            ans -= cval * val
        else:
            ans += cval * val

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    solve()