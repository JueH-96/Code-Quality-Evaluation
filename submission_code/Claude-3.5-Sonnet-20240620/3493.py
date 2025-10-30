class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        ones = s.count('1')
        zeros = n - ones
        
        if ones == 0 or zeros == 0:
            return 0
        
        operations = 0
        zero_count = 0
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                zero_count += 1
            else:
                operations += min(zero_count, ones - 1)
                ones -= 1
                if ones == 0:
                    break
        
        return operations