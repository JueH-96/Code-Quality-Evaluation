class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        from collections import Counter
        
        left = 0
        freq = Counter()
        max_len = 0
        
        for right in range(len(s)):
            freq[s[right]] += 1
            
            while freq[s[right]] > 2:
                freq[s[left]] -= 1
                if freq[s[left]] == 0:
                    del freq[s[left]]
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len