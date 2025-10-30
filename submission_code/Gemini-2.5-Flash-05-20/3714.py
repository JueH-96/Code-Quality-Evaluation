from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Precompute factorials and inverse factorials modulo MOD
        # Maximum value for 'n_val' in nCr(n_val, r_val) is n-1.
        # We need factorials and inverse factorials up to n (for nCr(n,r) where n_val can be n).
        # Array size n+1 to allow indexing up to n.
        
        # fact[i] will store i! % MOD
        fact = [1] * (n + 1)
        # inv_fact[i] will store (i!)^(-1) % MOD
        inv_fact = [1] * (n + 1)

        for i in range(1, n + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Calculate inverse factorial for n! using Fermat's Little Theorem
        # a^(p-2) % p is modular multiplicative inverse of a mod p
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)

        # Calculate inverse factorials for i from n-1 down to 0
        # inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        for i in range(n - 1, -1, -1):
            inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

        # Helper function to compute nCr % MOD (Binomial Coefficient)
        def nCr_mod_p(n_val: int, r_val: int) -> int:
            if r_val < 0 or r_val > n_val:
                return 0
            # C(n, 0) = C(n, n) = 1
            if r_val == 0 or r_val == n_val:
                return 1
            
            # nCr = n! / (r! * (n-r)!)
            # Modular inverse is used for division
            numerator = fact[n_val]
            denominator = (inv_fact[r_val] * inv_fact[n_val - r_val]) % MOD
            return (numerator * denominator) % MOD

        # Sort nums to easily identify elements to the left (smaller) and right (larger)
        nums.sort()

        total_sum = 0
        
        # Iterate through each number in the sorted array
        for i in range(n):
            current_num = nums[i]

            # Calculate how many subsequences have current_num as their minimum.
            # To be the minimum, current_num must be selected.
            # Other selected elements must be chosen from elements to its right (nums[i+1 ... n-1]).
            # There are (n - 1 - i) elements available to the right.
            # If we choose `p` additional elements from the right, the total subsequence length is `1 + p`.
            # This length must be at most `k`, so `1 + p <= k` => `p <= k - 1`.
            # `p` can range from 0 up to `min(k - 1, count_elements_right)`.
            count_elements_right = n - 1 - i
            sum_C_as_min = 0
            # The range for p is [0, min(k - 1, count_elements_right)]
            for p in range(min(k - 1, count_elements_right) + 1):
                sum_C_as_min = (sum_C_as_min + nCr_mod_p(count_elements_right, p)) % MOD

            # Calculate how many subsequences have current_num as their maximum.
            # To be the maximum, current_num must be selected.
            # Other selected elements must be chosen from elements to its left (nums[0 ... i-1]).
            # There are `i` elements available to the left.
            # If we choose `p` additional elements from the left, the total subsequence length is `1 + p`.
            # This length must be at most `k`, so `1 + p <= k` => `p <= k - 1`.
            # `p` can range from 0 up to `min(k - 1, count_elements_left)`.
            count_elements_left = i
            sum_C_as_max = 0
            # The range for p is [0, min(k - 1, count_elements_left)]
            for p in range(min(k - 1, count_elements_left) + 1):
                sum_C_as_max = (sum_C_as_max + nCr_mod_p(count_elements_left, p)) % MOD
            
            # The total contribution of current_num to the final sum:
            # `current_num` is added for each time it's a minimum and each time it's a maximum.
            # This correctly sums `(min_val + max_val)` for every valid subsequence.
            # For a subsequence of length 1 (e.g., `[x]`), `x` is both min and max,
            # so it's counted once by `sum_C_as_min` and once by `sum_C_as_max`, correctly adding `x+x`.
            contribution = (current_num * (sum_C_as_min + sum_C_as_max)) % MOD
            total_sum = (total_sum + contribution) % MOD

        return total_sum