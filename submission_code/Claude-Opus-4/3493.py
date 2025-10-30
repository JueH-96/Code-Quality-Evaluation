class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        ones_count = 0
        
        for i in range(len(s)):
            if s[i] == '1':
                ones_count += 1
            elif i > 0 and s[i-1] == '1':
                # We found a '10' pattern
                # All '1's to the left can eventually move past this group of '0's
                operations += ones_count
        
        return operations