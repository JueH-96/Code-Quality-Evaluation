class Solution:
    def minimumLength(self, s: str) -> int:
        left, right = 0, len(s) - 1
        
        while left < right and s[left] == s[right]:
            current_char = s[left]
            
            # Move the left pointer to the right as long as the character is the same
            while left <= right and s[left] == current_char:
                left += 1
            
            # Move the right pointer to the left as long as the character is the same
            while right >= left and s[right] == current_char:
                right -= 1
        
        return right - left + 1