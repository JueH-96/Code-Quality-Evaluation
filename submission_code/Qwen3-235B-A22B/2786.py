class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        max_len = 1
        left = 0
        count = 0
        
        for right in range(n):
            # Check if adding the current character creates a new consecutive pair
            if right > 0 and s[right] == s[right - 1]:
                count += 1
            
            # If the window has more than one pair, move the left pointer to the right
            while count > 1 and left < right:
                if s[left] == s[left + 1]:
                    count -= 1
                left += 1
            
            # Update the maximum length found
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len