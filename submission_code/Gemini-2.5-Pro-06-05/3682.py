class Solution:
    MOD = 10**9 + 7
    # n <= 10^5, so n-1 <= 99999. We need factorials up to 99999.
    # An array of size 100000 is sufficient (indices 0 to 99999).
    MAX_SIZE = 100000
    
    fact = [1] * MAX_SIZE
    inv_fact = [1] * MAX_SIZE
    _precomputed = False

    def _precompute(self):
        """
        Precomputes factorials and their modular inverses. This method is
        designed to run only once using a static flag.
        """
        if not Solution._precomputed:
            # Precompute factorials
            for i in range(1, Solution.MAX_SIZE):
                Solution.fact[i] = (Solution.fact[i-1] * i) % Solution.MOD
            
            # Precompute modular inverse of the largest factorial
            Solution.inv_fact[Solution.MAX_SIZE - 1] = pow(Solution.fact[Solution.MAX_SIZE - 1], Solution.MOD - 2, Solution.MOD)
            
            # Precompute other inverse factorials iteratively
            for i in range(Solution.MAX_SIZE - 2, -1, -1):
                Solution.inv_fact[i] = (Solution.inv_fact[i+1] * (i + 1)) % Solution.MOD
            
            Solution._precomputed = True

    def _combinations(self, n, k):
        """
        Calculates C(n, k) mod MOD using precomputed values.
        """
        if k < 0 or k > n:
            return 0
        
        numerator = Solution.fact[n]
        denominator_inv = (Solution.inv_fact[k] * Solution.inv_fact[n - k]) % Solution.MOD
        return (numerator * denominator_inv) % Solution.MOD

    def _power(self, base, exp):
        """
        Calculates (base^exp) % MOD using modular exponentiation.
        """
        return pow(base, exp, Solution.MOD)

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        
        # Ensure precomputation is done before any calculation.
        self._precompute()
        
        # The number of good arrays can be found by a combinatorial argument.
        # We first choose k positions for equal adjacent elements out of n-1
        # possible positions. This can be done in C(n-1, k) ways.
        num_placements = self._combinations(n - 1, k)
        
        # For each such choice, we count the ways to fill the array.
        # The first element has m choices.
        # Each of the k equal pairs (arr[i-1] == arr[i]) has 1 choice for arr[i].
        # Each of the n-1-k different pairs (arr[i-1] != arr[i]) has m-1 choices.
        # This gives m * (m-1)^(n-1-k) ways to choose the values.
        num_value_choices = (m * self._power(m - 1, n - 1 - k)) % self.MOD
        
        # The total number of good arrays is the product of these two counts.
        total_count = (num_placements * num_value_choices) % self.MOD
        
        return total_count