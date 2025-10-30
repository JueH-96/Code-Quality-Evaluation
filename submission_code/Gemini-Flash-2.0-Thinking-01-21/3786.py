class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        def get_cost(c1: str, c2: str) -> int:
            """Calculates the minimum operations to change character c1 to c2 (wrapping around)."""
            ord1 = ord(c1) - ord('a')
            ord2 = ord(c2) - ord('a')
            diff = abs(ord1 - ord2)
            return min(diff, 26 - diff)

        n = len(s)
        # dp[i][j][c] = max length of LPS in s[i...j] using AT MOST c operations
        # Initialize with 0.
        # The state space is N x N x (K+1)
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]

        # Base case: length 1 substrings
        # A single character s[i] is an LPS of length 1.
        # This requires 0 operations (just select the character).
        # Having more cost allowed doesn't change the fact that we can achieve length 1.
        for i in range(n):
            for cost in range(k + 1):
                dp[i][i][cost] = 1

        # Iterate over substring length (from 2 up to n)
        for length in range(2, n + 1):
            # Iterate over starting index (i)
            for i in range(n - length + 1):
                # Calculate ending index (j)
                j = i + length - 1
                # Iterate over allowed cost (c)
                # We iterate cost from 0 to k because the transition
                # dp[i+1][j-1][cost - match_cost] depends on states with less cost.
                for cost in range(k + 1):
                    # Option 1: Don't include s[j] in the subsequence.
                    # The LPS of s[i...j] using <= cost operations
                    # can be the LPS of s[i...j-1] using <= cost operations.
                    dp[i][j][cost] = max(dp[i][j][cost], dp[i][j-1][cost])

                    # Option 2: Don't include s[i] in the subsequence.
                    # The LPS of s[i...j] using <= cost operations
                    # can be the LPS of s[i+1...j] using <= cost operations.
                    dp[i][j][cost] = max(dp[i][j][cost], dp[i+1][j][cost])

                    # Option 3: Include both s[i] and s[j] as the outermost pair
                    # of the palindromic subsequence.
                    # To do this, we need to transform s[i] and s[j] into the same character.
                    # The minimum cost to make s[i] and s[j] match some character is get_cost(s[i], s[j]).
                    match_cost = get_cost(s[i], s[j])

                    # Check if we have enough cost to make s[i] and s[j] match.
                    if cost >= match_cost:
                        # We use `match_cost` operations for the ends s[i] and s[j].
                        # We have `cost - match_cost` operations left for the inner part s[i+1...j-1].

                        if length == 2:
                            # If length is 2, the substring is s[i...i+1].
                            # The inner part s[i+1...j-1] is s[i+1...i], which is an empty substring.
                            # The LPS length of an empty substring is 0.
                            # By matching s[i] and s[j], we get a palindrome of length 2.
                            dp[i][j][cost] = max(dp[i][j][cost], 2)
                        else: # length > 2, the inner part s[i+1...j-1] is a non-empty substring.
                            # The length from matching s[i] and s[j] is 2 plus the LPS length
                            # of the inner substring s[i+1...j-1] using the remaining cost.
                            dp[i][j][cost] = max(dp[i][j][cost], dp[i+1][j-1][cost - match_cost] + 2)

        # The final answer is the maximum length of an LPS for the entire string s[0...n-1]
        # using at most k operations. This value is stored in dp[0][n-1][k].
        # Since dp[i][j][c] stores the max length using *at most* c operations,
        # dp[0][n-1][k] automatically considers all possibilities with cost <= k.
        return dp[0][n-1][k]