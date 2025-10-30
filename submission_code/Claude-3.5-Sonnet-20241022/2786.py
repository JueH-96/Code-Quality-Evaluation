class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n <= 2:
            return n
            
        max_len = 1
        
        for i in range(n):
            pairs = 0
            for j in range(i+1, n):
                if s[j] == s[j-1]:
                    pairs += 1
                if pairs <= 1:
                    max_len = max(max_len, j-i+1)
                else:
                    break
                    
        return max_len