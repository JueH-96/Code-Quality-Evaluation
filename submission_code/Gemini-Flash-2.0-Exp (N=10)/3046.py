class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        ans = n
        for i in range(n):
            for j in range(i + 1, n):
                sub = num[i] + num[j]
                if sub in ("00", "25", "50", "75"):
                    ans = min(ans, n - j - 1 + i)
        
        
        for i in range(n):
            if num[i] == '0':
                ans = min(ans, n - 1)
        
        if '0' in num:
            ans = min(ans, n - num.rfind('0') - 1)
        
        return ans