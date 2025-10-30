class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        left = 0
        counts = {}
        max_len = 0
        
        for right, ch in enumerate(s):
            counts[ch] = counts.get(ch, 0) + 1
            
            # If this character occurs more than twice, shrink from the left
            while counts[ch] > 2:
                counts[s[left]] -= 1
                left += 1
            
            # Update max length
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len