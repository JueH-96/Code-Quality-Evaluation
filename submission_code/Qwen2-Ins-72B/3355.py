class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0] * n
        suffix = [0] * n
        prefix[0] = possible[0] * 2 - 1
        suffix[-1] = possible[-1] * 2 - 1
        
        for i in range(1, n):
            prefix[i] = prefix[i-1] + (possible[i] * 2 - 1)
        
        for i in range(n-2, -1, -1):
            suffix[i] = suffix[i+1] + (possible[i] * 2 - 1)
        
        for i in range(n-1):
            if prefix[i] > suffix[i+1]:
                return i + 1
        
        return -1