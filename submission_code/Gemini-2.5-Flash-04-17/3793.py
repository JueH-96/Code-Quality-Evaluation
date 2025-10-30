class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        m, n = len(s), len(t)

        # 1. Precompute LPS length for all substrings
        def compute_lps_substring(string):
            N = len(string)
            if N == 0:
                return [[]] # Handle empty string case for consistency, though constraints say N>=1

            # is_pal[i][j] = True if string[i..j] is palindrome
            is_pal = [[False] * N for _ in range(N)]
            # lps_val_sub[i][j] = length of LPS substring of string[i..j]
            lps_val_sub = [[0] * N for _ in range(N)]

            # Base case: single characters
            for i in range(N):
                is_pal[i][i] = True
                lps_val_sub[i][i] = 1

            # Fill for length 2 to N
            for l in range(2, N + 1):
                for i in range(N - l + 1):
                    j = i + l - 1

                    # Check if string[i..j] is a palindrome
                    if l == 2:
                        is_pal[i][j] = (string[i] == string[j])
                    else:
                        is_pal[i][j] = (string[i] == string[j]) and is_pal[i + 1][j - 1]

                    # Compute LPS length for string[i..j]
                    if is_pal[i][j]:
                        lps_val_sub[i][j] = l
                    else:
                        # The LPS substring of string[i..j] must be contained in string[i+1..j] or string[i..j-1].
                        # This DP structure correctly captures the max LPS *value* in the subproblem.
                        # Note: this is only true because we iterate outer loop by length l.
                        # For current length l, lps_val_sub[i+1][j] and lps_val_sub[i][j-1] would have been computed from smaller lengths.
                        # We need to be careful with indices when accessing previous values.
                        # When computing lps_val_sub[i][j], the required values lps_val_sub[i+1][j] (if i+1 <= j) and lps_val_sub[i][j-1] (if i <= j-1)
                        # correspond to substrings of length l-1 or less, which are already computed in previous iterations of the outer loop (by length).
                        val1 = lps_val_sub[i + 1][j] if i + 1 <= j else 0
                        val2 = lps_val_sub[i][j - 1] if i <= j - 1 else 0
                        lps_val_sub[i][j] = max(val1, val2)
            
            return lps_val_sub

        lps_s_sub = compute_lps_substring(s)
        lps_t_sub = compute_lps_substring(t)

        # 2. DP for concatenated palindromes
        # dp[i][j] = length of longest palindrome formed by substring from s[i..m-1] and substring from t[j..n-1].
        # i from m down to 0, j from n down to 0.
        # Table size (m+1) x (n+1).
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Base cases: one string contributes an empty substring
        # s_sub is empty (comes from s[m..m-1]). dp[m][j] considers substring from t[j..n-1].
        # The longest palindrome is the LPS substring of t[j..n-1].
        for j in range(n):
            dp[m][j] = lps_t_sub[j][n - 1]

        # t_sub is empty (comes from t[n..n-1]). dp[i][n] considers substring from s[i..m-1].
        # The longest palindrome is the LPS substring of s[i..m-1].
        for i in range(m):
            dp[i][n] = lps_s_sub[i][m - 1]
            
        # dp[m][n] = 0, already initialized

        # DP transitions
        # Iterate i from m-1 down to 0
        # Iterate j from n-1 down to 0
        # dp[i][j] is for s[i..m-1] and t[j..n-1].
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                # Option 1: The optimal s_sub does *not* start at s[i].
                # The s_sub must be from s[i+1..m-1]. The t_sub is from t[j..n-1].
                # Max length is dp[i+1][j].
                res1 = dp[i + 1][j]

                # Option 2: The optimal t_sub does *not* start at t[j].
                # The s_sub is from s[i..m-1]. The t_sub must be from t[j+1..n-1].
                # Max length is dp[i][j + 1]

                # Option 3: The optimal s_sub starts at s[i] AND the optimal t_sub ends at t[j].
                # This requires s[i] == t[j] to be the outermost pair.
                # If they match, length is 2 + length of longest palindrome from s[i+1..m-1] and t[j+1..n-1].
                # The length from s[i+1..m-1] and t[j+1..n-1] is exactly dp[i+1][j+1].
                res3 = 0
                if s[i] == t[j]:
                    # We can only extend if i+1..m-1 and j+1..n-1 are valid ranges.
                    # dp[i+1][j+1] already handles the base cases where one or both become empty.
                    res3 = 2 + dp[i + 1][j + 1]

                dp[i][j] = max(res1, res2, res3)

        # The answer is dp[0][0], considering s[0..m-1] and t[0..n-1].
        return dp[0][0]