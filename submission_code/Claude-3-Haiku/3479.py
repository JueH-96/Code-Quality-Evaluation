class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count = 0
        n = len(s)
        
        for i in range(n):
            zeros = 0
            ones = 0
            for j in range(i, n):
                if s[j] == '0':
                    zeros += 1
                else:
                    ones += 1
                if ones >= zeros ** 2:
                    count += n - j
                    break
        
        return count