class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        # Two-pointer sliding window approach
        max_len = 0
        left = 0
        count = {}
        
        for right in range(len(s)):
            ch = s[right]
            count[ch] = count.get(ch, 0) + 1
            
            # If any character count exceeds 2, shrink the window from the left
            while count[ch] > 2:
                count[s[left]] -= 1
                left += 1
            
            # The current window [left, right] is valid: update max length if needed.
            max_len = max(max_len, right - left + 1)
        
        return max_len