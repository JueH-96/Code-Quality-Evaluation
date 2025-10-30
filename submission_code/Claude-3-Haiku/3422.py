class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        a = [1] * n
        
        for _ in range(k):
            new_a = [a[0]]
            for i in range(1, n):
                new_a.append((new_a[-1] + a[i]) % MOD)
            a = new_a
        
        return a[-1]