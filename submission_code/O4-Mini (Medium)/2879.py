class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute for each length L (2..n) the list of divisors d where 1 <= d < L and L % d == 0
        divisors = [[] for _ in range(n+1)]
        for L in range(2, n+1):
            for d in range(1, L):
                if L % d == 0:
                    divisors[L].append(d)

        # cost[i][j]: minimal changes to turn s[i:j] into a semi-palindrome
        # only defined when j - i >= 2
        INF = 10**9
        cost = [[INF] * (n+1) for _ in range(n+1)]
        
        # Compute cost for every substring s[i:j] of length >= 2
        for i in range(n):
            for j in range(i+2, n+1):
                L = j - i
                best = INF
                # try each possible divisor d < L
                for d in divisors[L]:
                    # for this d, characters at positions i+m, i+m+d, i+m+2d, ... must form a palindrome
                    csum = 0
                    # handle each residue class m mod d
                    for m in range(d):
                        # collect chars in this class
                        chars = []
                        p = i + m
                        while p < j:
                            chars.append(s[p])
                            p += d
                        # cost to make this 'chars' list a palindrome
                        length = len(chars)
                        half = length // 2
                        for t in range(half):
                            if chars[t] != chars[length - 1 - t]:
                                csum += 1
                    if csum < best:
                        best = csum
                cost[i][j] = best

        # dp[i][t]: min cost to split prefix s[:i] into t semi-palindrome substrings
        # dp[0][0] = 0; answer is dp[n][k]
        dp = [[INF] * (k+1) for _ in range(n+1)]
        dp[0][0] = 0
        
        # fill dp
        for i in range(1, n+1):
            # at most t substrings in length i is i//2
            max_t = min(k, i // 2)
            for t in range(1, max_t + 1):
                # the last substring ends at i, starts at j, length >= 2
                # j must be >= 2*(t-1) so that prefix can hold t-1 substrings
                j_min = 2 * (t - 1)
                j_max = i - 2  # need at least length 2 for last part
                if j_max < j_min:
                    continue
                best = INF
                # try all possible start j of the last substring
                for j in range(j_min, j_max + 1):
                    prev = dp[j][t-1]
                    if prev >= INF:
                        continue
                    c = cost[j][i]
                    val = prev + c
                    if val < best:
                        best = val
                dp[i][t] = best
        
        return dp[n][k]