class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        freq = {}
        left = 0
        max_len = 0
        
        for right in range(len(s)):
            freq[s[right]] = freq.get(s[right], 0) + 1
            
            # Shrink the window if any character count exceeds 2
            while any(count > 2 for count in freq.values()):
                freq[s[left]] -= 1
                if freq[s[left]] == 0:
                    del freq[s[left]]
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len