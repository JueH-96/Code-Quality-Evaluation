class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        INF = 10**9
        
        # Precompute divisors for all lengths up to n
        divisors = [[] for _ in range(n+1)]
        for L in range(2, n+1):
            for d in range(1, int(L**0.5) + 1):
                if L % d == 0:
                    if d < L:
                        divisors[L].append(d)
                    dd = L // d
                    if dd != d and dd < L:
                        divisors[L].append(dd)
        
        # Precompute cost[i][j] = min changes to make s[i:j+1] a semi-palindrome
        cost = [[INF]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                L = j - i + 1
                best = INF
                # try every valid d
                for d in divisors[L]:
                    # the substring s[i..j], length L, step d
                    changes = 0
                    m = L // d  # number of items in each class
                    # for each residue class r in [0..d-1]
                    for r in range(d):
                        # for each pair in the class sequence
                        for p in range(m // 2):
                            c1 = s[i + r + p*d]
                            c2 = s[i + r + (m-1-p)*d]
                            if c1 != c2:
                                changes += 1
                    # update best for this substring
                    if changes < best:
                        best = changes
                cost[i][j] = best
        
        # dp[p][t]: min cost to partition s[:p] into t parts
        # we want dp[n][k]
        dp = [[INF]*(k+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        for p in range(1, n+1):
            for t in range(1, min(k, p//1) + 1):
                # end the t-th part at position p-1, start it at q .. p-1
                # previous parts cover s[:q]
                # at least t-1 chars to make t-1 parts of length >=1, but cost of length1 is INF
                # so effectively q >= 0
                for q in range(t-1, p):
                    c = cost[q][p-1]
                    if c < INF and dp[q][t-1] < INF:
                        val = dp[q][t-1] + c
                        if val < dp[p][t]:
                            dp[p][t] = val
        
        return dp[n][k]