class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        word_set = set(dictionary)
        n = len(s)
        if not word_set:
            return n
        
        max_len = max(len(word) for word in dictionary)
        
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            start_index = max(0, i - max_len)
            for j in range(start_index, i):
                if s[j:i] in word_set:
                    if dp[j] < dp[i]:
                        dp[i] = dp[j]
        return dp[n]