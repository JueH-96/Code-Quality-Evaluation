class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        """
        Finds the length of the longest palindromic subsequence of s that can be
        obtained after performing at most k operations.

        An operation is changing a character to its adjacent character in the alphabet,
        wrapping around ('a' after 'z', 'z' before 'a').

        Args:
            s: The input string.
            k: The maximum number of operations allowed.

        Returns:
            The length of the longest palindromic subsequence.
        """

        # Helper function to calculate the minimum operations to change c1 to c2
        # on a circular alphabet.
        def get_cost(c1, c2):
            ord1 = ord(c1) - ord('a')
            ord2 = ord(c2) - ord('a')
            diff = abs(ord1 - ord2)
            # The cost is the minimum distance on the circle
            return min(diff, 26 - diff)

        n = len(s)

        # dp[i][j][rem_k] will store the length of the longest palindromic
        # subsequence of the substring s[i...j] using at most rem_k operations.
        # Dimensions: n x n x (k + 1)
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]

        # Base case: Substrings of length 1 (i == j)
        # A single character is a palindrome of length 1. It costs 0 operations
        # to include this character in the subsequence. This is possible regardless
        # of the number of remaining operations.
        for i in range(n):
            for rem_k in range(k + 1):
                dp[i][i][rem_k] = 1

        # Iterate over substring lengths from 2 up to n
        for length in range(2, n + 1):
            # Iterate over starting index i
            for i in range(n - length + 1):
                # Calculate ending index j
                j = i + length - 1

                # Iterate over the allowed number of remaining operations
                for rem_k in range(k + 1):
                    # Option 1: The character s[j] is not included in the LPS.
                    # The problem reduces to finding the LPS of s[i...j-1]
                    # using at most rem_k operations.
                    dp[i][j][rem_k] = dp[i][j-1][rem_k]

                    # Option 2: The character s[i] is not included in the LPS.
                    # The problem reduces to finding the LPS of s[i+1...j]
                    # using at most rem_k operations.
                    dp[i][j][rem_k] = max(dp[i][j][rem_k], dp[i+1][j][rem_k])

                    # Option 3: The characters s[i] and s[j] are included as the
                    # outermost pair of the palindromic subsequence.
                    # To match s[i] and s[j], we need to spend a minimum number
                    # of operations on these two characters combined.
                    # The minimum total operations to make s[i] and s[j] equal
                    # to some character is get_cost(s[i], s[j]).
                    cost_ij = get_cost(s[i], s[j])

                    # If we have enough operations remaining to make s[i] and s[j] match:
                    if rem_k >= cost_ij:
                        # We use cost_ij operations for s[i] and s[j]. This pair contributes 2
                        # to the palindrome length.
                        # The remaining problem is to find the LPS of the inner substring s[i+1...j-1]
                        # using the remaining rem_k - cost_ij operations.
                        # dp[i+1][j-1][rem_k - cost_ij] gives this length.
                        # If length is 2 (j = i + 1), then i+1 > j-1. The range [i+1, j-1] is empty.
                        # The value dp[i+1][j-1][...] will be 0 due to the initialization,
                        # which is correct for an empty subsequence.
                        dp[i][j][rem_k] = max(dp[i][j][rem_k], 2 + dp[i+1][j-1][rem_k - cost_ij])

        # The final answer is the length of the longest palindromic subsequence
        # of the entire string s[0...n-1] using at most k operations.
        return dp[0][n-1][k]