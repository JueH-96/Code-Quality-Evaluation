class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                if (int(s[i]) % 2 == int(s[j]) % 2) and s[j] < s[i]:
                    s[i], s[j] = s[j], s[i]
                    return ''.join(s)
        
        return ''.join(s)