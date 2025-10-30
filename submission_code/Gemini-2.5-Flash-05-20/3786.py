class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        def get_cost(c1: str, c2: str) -> int:
            """Calculates the minimum operations to change c1 to c2 on a circular alphabet."""
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)

        N = len(s)
        
        # dp[i][j][rem_k] stores the length of the longest palindromic subsequence
        # for the substring s[i...j] using at most rem_k operations.
        # Initialize with 0s. This handles base cases for empty substrings (i > j) correctly.
        dp = [[[0] * (k + 1) for _ in range(N)] for _ in range(N)]

        # Base cases: Substrings of length 1 (i == j)
        # A single character forms a palindrome of length 1, costing 0 operations.
        # This holds true for any remaining budget, as 0 operations are used.
        for i in range(N):
            for rem_k_idx in range(k + 1):
                dp[i][i][rem_k_idx] = 1

        # Fill the DP table for increasing lengths of substrings
        # length ranges from 2 to N
        for length in range(2, N + 1):
            # i iterates from the beginning of the string up to the point
            # where a substring of `length` can still fit (i.e., i + length - 1 < N)
            for i in range(N - length + 1):
                j = i + length - 1 # j is the end index of the current substring s[i...j]
                
                # Iterate through all possible remaining budget values
                for rem_k_idx in range(k + 1):
                    # Option 1: s[i] and s[j] are NOT matched as a pair for the palindrome's ends.
                    # This means we either exclude s[i] or s[j] (or both) from the current pair.
                    # The value is the maximum of:
                    #   - LPS of s[i+1...j] with current rem_k_idx budget (excluding s[i])
                    #   - LPS of s[i...j-1] with current rem_k_idx budget (excluding s[j])
                    
                    # dp[i+1][j][rem_k_idx] is valid because i+1 <= j when length >= 2
                    dp[i][j][rem_k_idx] = max(dp[i][j][rem_k_idx], dp[i+1][j][rem_k_idx])
                    # dp[i][j-1][rem_k_idx] is valid because i <= j-1 when length >= 2
                    dp[i][j][rem_k_idx] = max(dp[i][j][rem_k_idx], dp[i][j-1][rem_k_idx])

                    # Option 2: Try to match s[i] and s[j] as a pair for the palindrome's ends.
                    # Calculate the cost to make s[i] and s[j] the same character.
                    cost = get_cost(s[i], s[j])
                    
                    # If we have enough budget to make them match
                    if rem_k_idx >= cost:
                        # If s[i] and s[j] are matched, they contribute 2 to the length.
                        # We then need to solve for the inner subsequence s[i+1...j-1]
                        # with the remaining budget rem_k_idx - cost.
                        
                        # inner_lps_length will be 0 if s[i+1...j-1] is an empty or invalid range (i+1 > j-1),
                        # because dp table is initialized with 0s.
                        inner_lps_length = dp[i+1][j-1][rem_k_idx - cost]
                        
                        dp[i][j][rem_k_idx] = max(dp[i][j][rem_k_idx], 2 + inner_lps_length)
        
        # The final answer is the LPS for the entire string s[0...N-1]
        # using at most k operations.
        return dp[0][N-1][k]