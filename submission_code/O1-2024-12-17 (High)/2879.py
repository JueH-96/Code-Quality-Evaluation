class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        import math

        n = len(s)
        INF = 10**9
        
        # Precompute all proper divisors d for each length L (1 <= d < L, L%d==0).
        # We'll store them in a list divisors[L].
        # For L < 2, there are no valid divisors (cannot form a semi-pal for length < 2).
        max_len = n
        divisors = [[] for _ in range(max_len + 1)]
        for L in range(2, max_len + 1):
            # 1 is always a divisor, and we'll check up to L-1
            # (d = L is not allowed because we need d < L).
            for d in range(1, L):
                if L % d == 0:
                    divisors[L].append(d)
            # Optional: sort divisors[L] so that 1 is at the front
            divisors[L].sort()
        
        # 1) Precompute palindromeCost[i][j]: minimum number of changes needed
        #    to make s[i..j] a (standard) palindrome.
        palindromeCost = [[0]*n for _ in range(n)]
        
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                if length == 2:
                    palindromeCost[i][j] = 0 if s[i] == s[j] else 1
                else:
                    palindromeCost[i][j] = palindromeCost[i+1][j-1]
                    if s[i] != s[j]:
                        palindromeCost[i][j] += 1
        
        # 2) Precompute costMatrix[i][j]: minimum changes to make s[i..j] a semi-palindrome.
        #    A string s[i..j] (length L=j-i+1 >=2) is semi-pal if there exists d (1<=d<L, L%d=0)
        #    such that columns mod d form palindromes. We'll take the min cost over all such d.
        costMatrix = [[INF]*n for _ in range(n)]
        
        for start in range(n):
            # length=1 is not a semi-pal, so costMatrix[start][start] = INF
            # We'll fill only for end>start.
            for end in range(start+1, n):
                L = end - start + 1
                # First, cost if we pick d=1 => must make entire substring a palindrome
                best_cost = palindromeCost[start][end]
                
                # Then, try other divisors d>1
                # divisors[L][0] should be 1, so skip that in the loop.
                for d in divisors[L][1:]:
                    # We'll compute the sum of costs needed to make each "column" a palindrome.
                    # There are d columns, each of length M = L//d.
                    M = L // d
                    total_col_cost = 0
                    for c in range(d):
                        # make s[start + c], s[start + c + d], ..., s[start + c + (M-1)*d] a palindrome
                        left = 0
                        right = M - 1
                        col_cost = 0
                        while left < right:
                            if s[start + c + left*d] != s[start + c + right*d]:
                                col_cost += 1
                            left += 1
                            right -= 1
                        total_col_cost += col_cost
                    
                    if total_col_cost < best_cost:
                        best_cost = total_col_cost
                
                costMatrix[start][end] = best_cost
        
        # 3) Use DP to partition s into exactly k substrings (all of length >=2)
        #    dp[i][p] = min total cost to form p semipal substrings covering s[0..i-1].
        dp = [[INF]*(k+1) for _ in range(n+1)]
        dp[0][0] = 0  # 0 substrings cover empty prefix with cost 0
        
        for p in range(1, k+1):
            for i in range(2, n+1):  # need at least length 2 to form a semipal
                best = dp[i][p]  # current best
                # try taking the last substring of length l >=2
                for l in range(2, i+1):
                    prev = dp[i-l][p-1]
                    if prev == INF:
                        continue
                    # cost for substring s[i-l..i-1]
                    c = costMatrix[i-l][i-1]
                    cand = prev + c
                    if cand < best:
                        best = cand
                dp[i][p] = best
        
        return dp[n][k]