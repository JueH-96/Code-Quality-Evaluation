class Solution:
    def stringCount(self, n: int) -> int:
        MOD = 10**9 + 7
        # Fast exponentiation for the bases
        p26 = pow(26, n, MOD)
        p25 = pow(25, n, MOD)
        p24 = pow(24, n, MOD)
        p23 = pow(23, n, MOD)
        # Powers for n-1 (n>=1 by problem constraint)
        p25_1 = pow(25, n-1, MOD)
        p24_1 = pow(24, n-1, MOD)
        p23_1 = pow(23, n-1, MOD)
        
        # Inclusion-Exclusion:
        # total = 26^n
        # A = no 'l'             => 25^n
        # B = no 't'             => 25^n
        # C = <2 'e'             => 25^n + n*25^(n-1)
        # AB = no 'l', no 't'    => 24^n
        # AC = no 'l', <2 'e'    => 24^n + n*24^(n-1)
        # BC = no 't', <2 'e'    => 24^n + n*24^(n-1)
        # ABC= no 'l','t',<2 'e' => 23^n + n*23^(n-1)
        #
        # Answer = 26^n
        #        - (|A|+|B|+|C|)
        #        + (|AB|+|AC|+|BC|)
        #        - |ABC|
        
        termA_B_C = (3 * p25 + n * p25_1) % MOD
        termAB_AC_BC = (3 * p24 + 2 * n * p24_1) % MOD
        termABC = (p23 + n * p23_1) % MOD
        
        ans = (p26 - termA_B_C + termAB_AC_BC - termABC) % MOD
        return ans