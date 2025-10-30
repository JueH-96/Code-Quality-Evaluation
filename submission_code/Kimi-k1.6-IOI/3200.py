class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        if n < 4:
            return 0
        
        term1 = pow(26, n, MOD)
        term2 = ( (75 + n) * pow(25, n-1, MOD) ) % MOD
        term3 = ( (72 + 2 * n) * pow(24, n-1, MOD) ) % MOD
        term4 = ( (23 + n) * pow(23, n-1, MOD) ) % MOD
        
        result = (term1 - term2 + term3 - term4) % MOD
        return result