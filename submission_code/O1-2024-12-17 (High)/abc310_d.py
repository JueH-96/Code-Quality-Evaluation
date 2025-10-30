def main():
    import sys
    sys.setrecursionlimit(10**7)

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    T = int(data[1])
    M = int(data[2])
    pairs = data[3:]
    
    # conflict_mask[v] will hold the bitmask of v itself and any vertex incompatible with v.
    conflict_mask = [(1 << i) for i in range(N)]
    idx = 0
    for _ in range(M):
        A = int(pairs[idx]) - 1
        B = int(pairs[idx+1]) - 1
        idx += 2
        conflict_mask[A] |= (1 << B)
        conflict_mask[B] |= (1 << A)
    
    # Precompute which subsets are "stable" (i.e., no two vertices in that subset conflict).
    # stable[s] = True if subset s is an independent set in the incompatibility graph.
    # Condition for subset s to be independent:
    #   For every vertex v in s, (s & conflict_mask[v]) == (1 << v)
    #   meaning no other vertices in s conflict with v.
    MAXS = (1 << N)
    stable = [False] * MAXS
    for s in range(MAXS):
        ok = True
        temp = s
        while temp and ok:
            v = (temp & -temp)            # lowest set bit of temp
            i = v.bit_length() - 1        # index of that bit
            if (s & conflict_mask[i]) != v:
                ok = False
            temp &= (temp - 1)
        stable[s] = ok
    
    # For each subset S, we will store all stable subsets I of S that contain
    # the smallest element in S.  This allows us to build partitions in a canonical order.
    stable_subsets_cache = [None] * MAXS
    
    def get_stable_subsets_containing(S):
        """Return all stable subsets I of S that contain the smallest set bit of S."""
        if S == 0:
            return []
        if stable_subsets_cache[S] is not None:
            return stable_subsets_cache[S]
        
        # Identify the smallest set bit in S
        low = (S & -S)   # bitmask with only the smallest set bit of S
        cand = S ^ low   # the remaining bits
        result = []
        
        # Enumerate all submasks x of cand, then I = x | low
        x = cand
        while True:
            I = x | low
            if stable[I]:
                result.append(I)
            if x == 0:
                break
            x = (x - 1) & cand
        
        stable_subsets_cache[S] = result
        return result
    
    # dp[S][t] = number of ways to partition the subset S (as a bitmask) into t stable (independent) subsets
    dp = [[-1]*(T+1) for _ in range(MAXS)]
    
    def popcount(x):
        return x.bit_count()  # Python 3.10+; otherwise use bin(x).count('1')
    
    def ways_to_partition(S, t):
        """Return the number of ways to partition subset S into t stable subsets."""
        if dp[S][t] != -1:
            return dp[S][t]
        if S == 0:
            dp[S][t] = 1 if t == 0 else 0
            return dp[S][t]
        
        pc = popcount(S)
        # If fewer bits than subsets, no way to place one vertex per subset.
        if pc < t:
            dp[S][t] = 0
            return 0
        
        # If we want to use exactly 1 subset, then it must be stable.
        if t == 1:
            dp[S][t] = 1 if stable[S] else 0
            return dp[S][t]
        
        # If we have exactly as many vertices as subsets, there's only one way:
        # each vertex goes to a separate subset.
        if pc == t:
            dp[S][t] = 1
            return 1
        
        # Otherwise, pick a stable subset containing the smallest set bit of S,
        # remove it, and recurse on the remainder with t-1.
        answer = 0
        for subset in get_stable_subsets_containing(S):
            answer += ways_to_partition(S & ~subset, t - 1)
        
        dp[S][t] = answer
        return answer
    
    full_set = (1 << N) - 1
    print(ways_to_partition(full_set, T))

# Don't forget to call main()
if __name__ == "__main__":
    main()