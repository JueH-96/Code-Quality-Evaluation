class Solution:
    def minimumLength(self, s: str) -> int:
        left = 0
        right = len(s) - 1
        
        while left < right and s[left] == s[right]:
            curr = s[left]
            
            # Skip all matching characters from left
            while left <= right and s[left] == curr:
                left += 1
                
            # Skip all matching characters from right
            while left <= right and s[right] == curr:
                right -= 1
                
        return right - left + 1