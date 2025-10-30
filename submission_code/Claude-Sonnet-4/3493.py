class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        ones_count = 0
        
        for i in range(len(s)):
            if s[i] == '1':
                ones_count += 1
            else:  # s[i] == '0'
                # If previous character was '1', we're starting a new block of 0's
                # All 1's seen so far can move past this block of 0's
                if i > 0 and s[i-1] == '1':
                    operations += ones_count
        
        return operations