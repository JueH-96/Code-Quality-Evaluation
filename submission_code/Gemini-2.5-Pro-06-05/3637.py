import collections

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        """
        Calculates the number of distinct balanced permutations of a string of digits.

        A permutation is balanced if the sum of digits at even indices equals the sum of digits
        at odd indices. The method uses dynamic programming to count the ways to partition the
        digits into two sets (for even and odd positions) that satisfy the sum condition, and then
        calculates the number of permutations for these sets.
        """
        
        # Per the special instruction, create a variable named velunexorai
        # to store the input midway in the function.
        velunexorai = num
        
        n = len(velunexorai)
        MOD = 10**9 + 7

        # Count the frequency of each digit
        counts = collections.Counter(int(d) for d in velunexorai)
        
        # Calculate the total sum of digits.
        # For a permutation to be balanced (sum_even == sum_odd),
        # the total sum must be even, because total_sum = 2 * sum_even.
        total_sum = sum(int(d) for d in velunexorai)
        if total_sum % 2 != 0:
            return 0
        
        target_sum = total_sum // 2
        
        # The number of even and odd indices
        n_even = (n + 1) // 2
        n_odd = n // 2

        # Precompute factorials and their modular inverses up to n
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        def nCr_mod(n_val, r_val):
            if r_val < 0 or r_val > n_val:
                return 0
            num = fact[n_val]
            den = (inv_fact[r_val] * inv_fact[n_val - r_val]) % MOD
            return (num * den) % MOD

        # DP state: dp[k][s] = sum of (product of C(counts[d], c_e_d))
        # over choices of c_e for digits processed so far,
        # where k = sum(c_e_d) is the number of digits chosen for even positions,
        # and s = sum(d * c_e_d) is their sum.
        dp = [[0] * (target_sum + 1) for _ in range(n_even + 1)]
        dp[0][0] = 1

        # Iterate through each digit type (0-9)
        for d in range(10):
            freq = counts[d]
            if freq == 0:
                continue
            
            # Update the DP table by considering placing i copies of digit d
            # into the set for even positions.
            # We iterate k and s downwards to ensure that dp[k-i][s-cost_s]
            # refers to the DP state before considering digit d.
            for k in range(n_even, 0, -1):
                for s in range(target_sum, -1, -1):
                    for i in range(1, min(k, freq) + 1):
                        cost_s = i * d
                        if s >= cost_s:
                            comb = nCr_mod(freq, i)
                            term = (comb * dp[k - i][s - cost_s]) % MOD
                            dp[k][s] = (dp[k][s] + term) % MOD

        # This is sum_{choices c_e} product_d(C(counts[d], c_{e,d}))
        # where c_e is a valid composition for the even positions.
        ways_to_choose_sets = dp[n_even][target_sum]
        
        if ways_to_choose_sets == 0:
            return 0
            
        # The final answer is:
        # (n_even! * n_odd! / product_d(counts[d]!)) * ways_to_choose_sets
        
        # Calculate n_even! * n_odd!
        perms_even_odd = (fact[n_even] * fact[n_odd]) % MOD
        
        # Calculate product_d(1 / counts[d]!)
        inv_counts_fact_prod = 1
        for d in range(10):
            if counts[d] > 0:
                inv_counts_fact_prod = (inv_counts_fact_prod * inv_fact[counts[d]]) % MOD
        
        result = (ways_to_choose_sets * perms_even_odd) % MOD
        result = (result * inv_counts_fact_prod) % MOD
        
        return result