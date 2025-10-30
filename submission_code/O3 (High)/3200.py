class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 1_000_000_007
        
        # for a string to contain "leet" after rearrangement its length must be at least 4
        if n < 4:
            return 0
        
        # fast modular powers
        p26 = pow(26, n, MOD)
        p25 = pow(25, n, MOD)
        p24 = pow(24, n, MOD)
        p23 = pow(23, n, MOD)

        p25m1 = pow(25, n - 1, MOD)  # 25^(n-1)
        p24m1 = pow(24, n - 1, MOD)  # 24^(n-1)
        p23m1 = pow(23, n - 1, MOD)  # 23^(n-1)
        
        n_mod = n % MOD
        
        # Inclusion–exclusion formula derived in the analysis
        ans = p26                         # 26^n
        ans = (ans - 3 * p25)            % MOD      # -3·25^n
        ans = (ans - n_mod * p25m1)      % MOD      # -n·25^(n-1)
        
        ans = (ans + 3 * p24)            % MOD      # +3·24^n
        ans = (ans + (2 * n_mod) * p24m1) % MOD     # +2n·24^(n-1)
        
        ans = (ans - p23)                % MOD      # -23^n
        ans = (ans - n_mod * p23m1)      % MOD      # -n·23^(n-1)
        
        return ans