class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        
        if n == 1:
            return 1
        
        a = [1] * n
        
        for _ in range(k):
            new_a = [0] * n
            new_a[0] = a[0]
            for i in range(1,n):
                new_a[i] = (sum(a[:i+1])) % mod
            a = new_a

        return a[n-1]