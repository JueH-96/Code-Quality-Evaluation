class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        ones = []
        for i in range(n):
            if s[i] == '1':
                ones.append(i)
        
        ans = 0
        for i in range(len(ones)):
            ans += ones[i] - (n - len(ones) + i)
        return ans