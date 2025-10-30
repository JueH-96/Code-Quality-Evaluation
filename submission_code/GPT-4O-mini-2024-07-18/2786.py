class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        max_length = 0
        n = len(s)
        
        for i in range(n):
            count = 0
            has_pair = False
            
            for j in range(i, n):
                if j > 0 and s[j] == s[j - 1]:
                    if has_pair:
                        break
                    has_pair = True
                
                count += 1
                max_length = max(max_length, count)
        
        return max_length