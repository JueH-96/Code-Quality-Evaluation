class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        ones_count = 0
        
        for char in s:
            if char == '1':
                ones_count += 1
            else:  # char == '0'
                steps += ones_count
        
        return steps