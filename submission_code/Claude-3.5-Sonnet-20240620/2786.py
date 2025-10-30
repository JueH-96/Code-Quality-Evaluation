class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n <= 2:
            return n
        
        max_length = 0
        start = 0
        last_pair = -1
        
        for i in range(1, n):
            if s[i] == s[i-1]:
                if last_pair != -1:
                    start = last_pair + 1
                last_pair = i - 1
            
            max_length = max(max_length, i - start + 1)
        
        return max_length