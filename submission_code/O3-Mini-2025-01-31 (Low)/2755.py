from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)

        # dp[n] = 0 because no extra characters remain if we are beyond last index
        for i in range(n - 1, -1, -1):
            # Assume we skip character at i, thus extra 1 and then best solution from i+1
            dp[i] = dp[i + 1] + 1
            for word in dictionary:
                if s.startswith(word, i):
                    j = i + len(word)
                    # Using this word, no extra char for these indices, then use best from j
                    dp[i] = min(dp[i], dp[j])
        return dp[0]
        
# You can include some basic tests below
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    s1 = "leetscode"
    dictionary1 = ["leet","code","leetcode"]
    print(sol.minExtraChar(s1, dictionary1))  # Expected output: 1

    # Example 2:
    s2 = "sayhelloworld"
    dictionary2 = ["hello","world"]
    print(sol.minExtraChar(s2, dictionary2))  # Expected output: 3