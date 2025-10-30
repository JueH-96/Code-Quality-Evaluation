class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [0] * (n + 1)  # dp[0] is 0
        
        for i in range(1, n + 1):
            # Initialize with the case where current character is extra
            dp[i] = dp[i - 1] + 1
            
            # Check all words in the dictionary
            for word in dictionary:
                l = len(word)
                if l <= i:
                    # Check if the substring ending at i matches the word
                    if s[i - l:i] == word:
                        if dp[i - l] < dp[i]:
                            dp[i] = dp[i - l]
        
        return dp[n]