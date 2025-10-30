class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 1
        
        # Sliding window approach
        l = 0
        count = 0  # number of consecutive repeated pairs in the current window
        max_len = 1
        
        for r in range(n):
            if r > 0 and s[r] == s[r - 1]:
                count += 1
            
            while count > 1:
                if s[l] == s[l + 1]:
                    count -= 1
                l += 1
            
            max_len = max(max_len, r - l + 1)
        
        return max_len