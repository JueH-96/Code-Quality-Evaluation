class Solution:
    def maxOperations(self, s: str) -> int:
        s = list(s)
        count = 0
        n = len(s)
        
        for i in range(n - 1):
            if s[i] == '1' and s[i+1] == '0':
                j = i
                while j < n - 1 and s[j+1] == '0':
                    s[j], s[j+1] = s[j+1], s[j]
                    j += 1
                count += 1
        return count