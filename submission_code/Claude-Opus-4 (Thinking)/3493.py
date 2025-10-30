class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ones_count = 0
        operations = 0
        
        for i in range(n - 1, -1, -1):
            if s[i] == '1':
                ones_count += 1
            else:  # s[i] == '0'
                # Check if this is the rightmost '0' in a contiguous block
                if i == n - 1 or s[i + 1] == '1':
                    operations += ones_count
        
        return operations