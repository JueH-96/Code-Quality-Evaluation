class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        count_of_ones = 0
        
        for char in s:
            if char == '1':
                count_of_ones += 1
            else:  # char == '0'
                steps += count_of_ones
        
        return steps