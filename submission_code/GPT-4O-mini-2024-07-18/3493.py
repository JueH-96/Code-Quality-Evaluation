class Solution:
    def maxOperations(self, s: str) -> int:
        count_ones = 0
        operations = 0
        
        for char in s:
            if char == '1':
                count_ones += 1
            elif char == '0' and count_ones > 0:
                operations += 1
                count_ones -= 1
        
        return operations