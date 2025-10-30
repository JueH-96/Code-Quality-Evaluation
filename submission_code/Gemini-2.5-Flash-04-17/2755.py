from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # Convert dictionary to a set for O(1) average time lookups
        word_set = set(dictionary)

        # dp[i] will store the minimum extra characters in the suffix s[i:]
        # Initialize with a value representing the maximum possible extra characters (n)
        dp = [0] * (n + 1)

        # Iterate backwards through the string
        # dp[n] is 0 (empty suffix has no extra characters)
        for i in range(n - 1, -1, -1):
            # Option 1: The character s[i] is an extra character
            # Minimum extra for s[i:] is 1 (for s[i]) + min extra for s[i+1:]
            dp[i] = dp[i+1] + 1

            # Option 2: s[i] is the start of a dictionary word
            # Check all possible end positions j for a word starting at i
            # The substring is s[i:j+1] (from index i up to j)
            for j in range(i, n):
                substring = s[i : j+1]
                if substring in word_set:
                    # If s[i:j+1] is a dictionary word, we can use it.
                    # These characters s[i]...s[j] contribute 0 extra characters.
                    # The total extra characters would be the minimum extra characters
                    # for the remaining suffix s[j+1:] (which is dp[j+1]).
                    dp[i] = min(dp[i], dp[j+1])

        # The minimum extra characters for the entire string s[0:] is dp[0]
        return dp[0]