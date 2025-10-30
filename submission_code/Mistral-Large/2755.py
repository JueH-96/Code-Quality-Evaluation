from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        # Convert the dictionary to a set for O(1) lookups
        word_set = set(dictionary)

        # Initialize a DP array where dp[i] represents the minimum number of extra characters
        # when considering the substring s[0:i]
        n = len(s)
        dp = [0] * (n + 1)

        # Iterate over the string
        for i in range(1, n + 1):
            # Initialize the minimum extra characters for the current position
            dp[i] = dp[i - 1] + 1

            # Check all possible substrings ending at position i
            for j in range(i):
                if s[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j])

        return dp[n]