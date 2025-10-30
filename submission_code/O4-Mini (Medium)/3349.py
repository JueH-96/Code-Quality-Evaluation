class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        n = len(s)
        counts = [0] * 26
        left = 0
        max_len = 0
        
        for right, ch in enumerate(s):
            idx = ord(ch) - ord('a')
            counts[idx] += 1
            
            # If this character occurs more than twice, move left pointer
            while counts[idx] > 2:
                counts[ord(s[left]) - ord('a')] -= 1
                left += 1
            
            max_len = max(max_len, right - left + 1)
        
        return max_len