class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        max_len = 0
        counts = {}
        
        for right in range(len(s)):
            counts[s[right]] = counts.get(s[right], 0) + 1
            
            while counts[s[right]] > 2:
                counts[s[left]] -= 1
                if counts[s[left]] == 0:
                    del counts[s[left]]
                left += 1
                
            max_len = max(max_len, right - left + 1)
        
        return max_len