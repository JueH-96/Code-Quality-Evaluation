class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        dict_set = set(dictionary)
        n = len(s)
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1  # Initialize assuming current character is extra
            for j in range(i):
                if s[j:i] in dict_set:
                    if dp[j] < dp[i]:
                        dp[i] = dp[j]
        
        return dp[n]