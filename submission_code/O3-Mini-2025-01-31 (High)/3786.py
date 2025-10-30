class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        # Precompute the cost to change one letter to another.
        # cost(a, b) = min(|a-b|, 26 - |a-b|)
        cost = [[0] * 26 for _ in range(26)]
        for a in range(26):
            for b in range(26):
                diff = abs(a - b)
                cost[a][b] = diff if diff <= 13 else 26 - diff

        # For any two letters a and b (represented as indices 0..25),
        # compute the minimum cost needed to turn both into the same letter.
        # This will be used when we want to "pair" two chosen letters.
        pair_cost = [[0] * 26 for _ in range(26)]
        for a in range(26):
            for b in range(26):
                best = 10**9
                for x in range(26):
                    cst = cost[a][x] + cost[b][x]
                    if cst < best:
                        best = cst
                pair_cost[a][b] = best

        n = len(s)
        # k_plus is just k+1 (we use indices 0...k)
        k_plus = k + 1
        # dp[i][j] will be an array of length (k+1) where
        # dp[i][j][r] = maximum palindromic subsequence length you can get
        # from the substring s[i...j] using at most r operations.
        dp = [[None] * n for _ in range(n)]
        
        # Base case: a single letter is a palindrome of length 1 (no operations needed)
        for i in range(n):
            dp[i][i] = [1] * k_plus
        
        a_ord = ord('a')
        # Build the dp table by increasing interval length.
        # length is the length of the current substring.
        for length in range(2, n+1):
            for i in range(n - length + 1):
                j = i + length - 1
                # For computing dp[i][j] we use:
                #   dp[i+1][j] (skipping s[i]),
                #   dp[i][j-1] (skipping s[j]),
                # and if we can "pair" s[i] and s[j] (by spending some operations),
                #   we use 2 + dp[i+1][j-1] with the remaining cost.
                left = dp[i+1][j]   # substring s[i+1...j]
                right = dp[i][j-1]  # substring s[i...j-1]
                diag = dp[i+1][j-1] if i+1 <= j-1 else None  # substring s[i+1...j-1]; if not there, use 0.
                
                # Compute the cost needed to make s[i] and s[j] equal.
                c1 = ord(s[i]) - a_ord
                c2 = ord(s[j]) - a_ord
                cost_pair = pair_cost[c1][c2]
                
                # We are going to compute an array "res" for the current dp[i][j]
                res = [0] * k_plus
                for r in range(k_plus):
                    # Option 1: Skip s[i] (or use dp[i+1][j])
                    cand = left[r]
                    # Option 2: Skip s[j] (or use dp[i][j-1])
                    if right[r] > cand:
                        cand = right[r]
                    # Option 3: If we have enough budget to “pair” s[i] and s[j],
                    # then we add 2 and add the best from dp[i+1][j-1] with cost r - cost_pair.
                    if r >= cost_pair:
                        diag_val = diag[r - cost_pair] if diag is not None else 0
                        pairing = 2 + diag_val
                        if pairing > cand:
                            cand = pairing
                    res[r] = cand
                # Enforce monotonicity: with more operations we cannot get a smaller answer.
                for r in range(1, k_plus):
                    if res[r] < res[r-1]:
                        res[r] = res[r-1]
                dp[i][j] = res

        return dp[0][n-1][k]