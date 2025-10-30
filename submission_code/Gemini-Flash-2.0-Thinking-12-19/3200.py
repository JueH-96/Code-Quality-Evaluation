class Solution:
    def stringCount(self, n: int) -> int:
        if n < 4:
            return 0
        mod = 10**9 + 7
        
        def power(base, exp):
            res = 1
            base %= mod
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % mod
                exp >>= 1
                base = (base * base) % mod
            return res
            
        def inverse(n):
            return power(n, mod - 2)
            
        p26n = power(26, n)
        p25n = power(25, n)
        p24n = power(24, n)
        p23n = power(23, n)
        
        p25nm1 = (p25n * inverse(25)) % mod
        p24nm1 = (p24n * inverse(24)) % mod
        p23nm1 = (p23n * inverse(23)) % mod
        
        term1 = p26n
        term2 = (3 * p25n) % mod
        term3 = (n * p25nm1) % mod
        term4 = (3 * p24n) % mod
        term5 = (2 * n * p24nm1) % mod
        term6 = p23n
        term7 = (n * p23nm1) % mod
        
        result = (term1 - term2 - term3 + term4 + term5 - term6 - term7) % mod
        return result