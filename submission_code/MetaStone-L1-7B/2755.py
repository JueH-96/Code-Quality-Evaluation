class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        word_set = set(dictionary)
        max_word_len = max(len(word) for word in word_set) if word_set else 0
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            current_min = dp[i-1] + 1  # Option: take the last character as extra
            
            start = max(0, i - max_word_len)
            for j in range(start, i):
                substr = s[j:i]
                if substr in word_set:
                    len_word = i - j
                    extra = len_word - len_word  # This will always be 0
                    if dp[j] + extra < current_min:
                        current_min = dp[j] + extra
            dp[i] = current_min
        
        return dp[n]