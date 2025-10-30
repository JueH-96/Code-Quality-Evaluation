class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        ones_count = 0
        
        for char in s:
            if char == '1':
                ones_count += 1
            elif ones_count > 0:
                operations += 1
                ones_count -= 1
        
        return operations