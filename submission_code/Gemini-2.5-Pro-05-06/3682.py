class Solution:
    MOD = 10**9 + 7
    
    # Max value for N_val in NCr is n-1. Max n = 10^5, so max N_val = 10^5 - 1.
    # Arrays need to be size 10^5 to store indices 0 to (10^5 - 1).
    _MAX_N_COMB_STORAGE = 10**5 
    _fact = [1] * _MAX_N_COMB_STORAGE
    _invFact = [1] * _MAX_N_COMB_STORAGE

    @classmethod
    def _init_combinatorics_once(cls):
        """Performs one-time precomputation of factorials and inverse factorials."""
        
        # Calculate factorials: _fact[i] = i! % MOD
        # _fact[0] is already 1.
        for i in range(1, cls._MAX_N_COMB_STORAGE):
            cls._fact[i] = (cls._fact[i-1] * i) % cls.MOD

        # Calculate inverse factorials: _invFact[i] = (i!)^(-1) % MOD
        # We use Fermat's Little Theorem: a^(p-2) = a^(-1) mod p for prime p.
        # _invFact[N] = (_fact[N])^(MOD-2) % MOD
        # Then, _invFact[i-1] = _invFact[i] * i % MOD
        
        # Handle _MAX_N_COMB_STORAGE = 0 or 1, though it's fixed at 10^5.
        # If _MAX_N_COMB_STORAGE = 0, arrays are empty. Accesses would fail.
        # If _MAX_N_COMB_STORAGE = 1 (max N_val = 0, e.g., max n=1 for combinations):
        #   _fact = [1], _invFact = [1]
        #   Factorial loop range(1,1) does not run. _fact[0] remains 1.
        #   _invFact[0] = pow(_fact[0], MOD-2, MOD) = pow(1, MOD-2, MOD) = 1.
        #   Inverse factorial loop range(_MAX_N_COMB_STORAGE - 2 = -1, down to -1) does not run.
        #   This setup (_fact[0]=1, _invFact[0]=1) is correct for C(0,0).
        
        if cls._MAX_N_COMB_STORAGE > 0:
             cls._invFact[cls._MAX_N_COMB_STORAGE - 1] = pow(cls._fact[cls._MAX_N_COMB_STORAGE - 1], cls.MOD - 2, cls.MOD)
        
        # Loop from _MAX_N_COMB_STORAGE - 2 down to 0
        for i in range(cls._MAX_N_COMB_STORAGE - 2, -1, -1): 
            cls._invFact[i] = (cls._invFact[i+1] * (i+1)) % cls.MOD
            
    # Perform precomputation when the class is defined. This ensures it runs only once.
    _init_combinatorics_once()

    def _nCr_mod_p(self, N_val: int, R_val: int) -> int:
        """Computes C(N_val, R_val) % MOD using precomputed values."""
        # Based on problem constraints (0 <= k <= n-1):
        # N_val = n-1, R_val = k. So R_val is always in [0, N_val].
        # Max N_val = 10^5 - 1, which is < _MAX_N_COMB_STORAGE.
        # So, indices N_val, R_val, N_val-R_val are valid for _fact and _invFact arrays.
        if R_val < 0 or R_val > N_val: # Should not be hit with problem constraints
            return 0
            
        numerator = self._fact[N_val]
        inv_R_fact = self._invFact[R_val]
        inv_N_minus_R_fact = self._invFact[N_val - R_val]
        
        denominator = (inv_R_fact * inv_N_minus_R_fact) % self.MOD
        
        return (numerator * denominator) % self.MOD

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        # Number of adjacent pairs is n-1.
        # We choose k of these to be equal pairs: C(n-1, k) ways.
        # The first element arr[0] has m choices.
        # The k equal pairs contribute a factor of 1^k = 1.
        # The (n-1)-k unequal pairs contribute a factor of (m-1)^((n-1)-k).
        # Total = m * C(n-1, k) * (m-1)^((n-1)-k)

        num_pairs = n - 1 # This is N for C(N,k)
        
        # Calculate C(num_pairs, k)
        combinations_val = self._nCr_mod_p(num_pairs, k)
        
        # Calculate (m-1)^((num_pairs)-k)
        exponent_val = num_pairs - k
        # Python's pow(base, exp, mod) handles base = 0 correctly:
        # pow(0, 0, mod) returns 1.
        # pow(0, positive_exp, mod) returns 0.
        # This correctly handles the m=1 case.
        term_m_minus_1 = pow(m - 1, exponent_val, self.MOD)
        
        # Multiply parts together, taking modulo at each step.
        ans = (m * combinations_val) % self.MOD
        ans = (ans * term_m_minus_1) % self.MOD
        
        return ans