class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        maximum = 0
        
        for char in s:
            if char == 'E':
                current += 1
                maximum = max(maximum, current)
            else:  # char == 'L'
                current -= 1
        
        return maximum