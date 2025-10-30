class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        """
        Dynamic programming on the classical identity:
            LPS(s) = LCS(s , reverse(s))
        We add one more dimension – the number of operations already spent – and
        let every match between characters cost the smallest number of single–step
        moves that are necessary to make the two letters equal on a cyclic alphabet
        of length 26.

        dp[i][j][c]  (implemented with rolling rows to save memory)
            = the best common–subsequence length that can be obtained from
              s[0..i) and rev[0..j) spending exactly c operations.

        Transition (usual LCS choices + paying the matching cost):

            1) skip s[i-1]      -> dp[i-1][j][c]
            2) skip rev[j-1]    -> dp[i][j-1][c]
            3) match them
                  need   cost = dist(s[i-1], rev[j-1])
                  if c >= cost:
                        dp[i-1][j-1][c-cost] + 1

        Finally  answer = max_{c<=k} dp[n][n][c]

        Complexity
            time  : O(n² · k)  = 200·200·200 ≈ 8·10⁶ – fine
            memory: two rows  (n+1)·(k+1)      ≈ 40 000 integers ≈ 1 MB
        """
        n = len(s)
        rev = s[::-1]

        # distance on the cyclic alphabet
        def dist(a: str, b: str) -> int:
            d = abs(ord(a) - ord(b))
            return d if d <= 13 else 26 - d     # 13 = 26//2

        # dp_prev[j][c] – already processed prefix of s (up to i-1),
        #                 first j letters of rev, cost c
        dp_prev = [[0] * (k + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp_cur = [[0] * (k + 1) for _ in range(n + 1)]
            a = s[i - 1]
            for j in range(1, n + 1):
                b = rev[j - 1]
                cost_pair = dist(a, b)

                prev_col = dp_cur[j - 1]   # skipping rev[j-1]
                above    = dp_prev[j]      # skipping s[i-1]
                diag     = dp_prev[j - 1]  # match possibility

                cur = dp_cur[j]
                for c in range(k + 1):
                    best = above[c]                 # option 1
                    if prev_col[c] > best:          # option 2
                        best = prev_col[c]

                    if c >= cost_pair:              # option 3 (match them)
                        val = diag[c - cost_pair] + 1
                        if val > best:
                            best = val
                    cur[c] = best
            dp_prev = dp_cur                       # roll row

        return max(dp_prev[-1])                    # best length with ≤ k ops