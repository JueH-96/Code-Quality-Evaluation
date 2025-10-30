class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = [i for i in range(n) if s1[i] != s2[i]]
        
        if len(diff) % 2 != 0:
            return -1
        
        operations = 0
        i = 0
        while i < len(diff):
            if diff[i + 1] - diff[i] == 1:
                operations += 1
                i += 2
            else:
                operations += x
                i += 2
        
        return operations