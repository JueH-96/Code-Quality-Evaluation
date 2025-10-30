class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        a = k + n - 1
        b = n - 1
        
        numerator = 1
        for i in range(b):
            numerator = numerator * (a - i) % MOD
        
        denominator = 1
        for i in range(1, b + 1):
            denominator = denominator * i % MOD
        
        return numerator * pow(denominator, MOD - 2, MOD) % MOD