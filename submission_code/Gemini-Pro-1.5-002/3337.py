class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = len(s)
        count = 0
        indices = []
        for i in range(n):
            if s[i] == c:
                indices.append(i)
        
        ans = 0
        for i in range(len(indices)):
            for j in range(i, len(indices)):
                ans += 1
        return ans