class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 1000000007
        # Compute powers modulo MOD
        pow26_n = pow(26, n, MOD)
        pow25_n = pow(25, n, MOD)
        pow25_nm1 = pow(25, n - 1, MOD)
        pow24_n = pow(24, n, MOD)
        pow24_nm1 = pow(24, n - 1, MOD)
        pow23_n = pow(23, n, MOD)
        pow23_nm1 = pow(23, n - 1, MOD)
        
        # Compute part1: 3 * 25^n + n * 25^{n-1}
        part1 = ((3 * pow25_n) % MOD + (n * pow25_nm1 % MOD)) % MOD
        
        # Compute part2: 3 * 24^n + 2 * n * 24^{n-1}
        part2 = ((3 * pow24_n) % MOD + ((2 * n * pow24_nm1) % MOD)) % MOD
        
        # Compute part3: 23^n + n * 23^{n-1}
        part3 = ((pow23_n) % MOD + (n * pow23_nm1 % MOD)) % MOD
        
        # Compute bad: part1 - part2 + part3
        bad = (part1 - part2 + part3) % MOD
        bad = (bad + MOD) % MOD  # Ensure non-negative
        
        # Compute good: 26^n - bad
        good = (pow26_n - bad) % MOD
        good = (good + MOD) % MOD  # Ensure non-negative
        
        return good