from typing import List

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        # dp[i] = minimum extra characters for substring s[i:]
        dp = [0] * (n + 1)
        # Base case: empty suffix has 0 extra characters
        dp[n] = 0
        
        # Preprocess dictionary into a set for O(1) membership checks
        dict_set = set(dictionary)
        
        for i in range(n - 1, -1, -1):
            # Option 1: treat s[i] as an extra character
            dp[i] = 1 + dp[i + 1]
            
            # Option 2: try to match any dictionary word starting at i
            # Since max word length is 50 and n <= 50, we can bound the scan
            for word in dict_set:
                wl = len(word)
                if i + wl <= n and s[i:i + wl] == word:
                    dp[i] = min(dp[i], dp[i + wl])
        
        return dp[0]

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minExtraChar("leetscode", ["leet","code","leetcode"]))  # Output: 1
    print(sol.minExtraChar("sayhelloworld", ["hello","world"]))       # Output: 3