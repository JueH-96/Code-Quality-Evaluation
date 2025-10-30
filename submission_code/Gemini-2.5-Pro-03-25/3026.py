import sys
# Setting a higher recursion depth might not be strictly necessary for this iterative solution,
# but can be included as a safeguard in competitive programming environments.
# sys.setrecursionlimit(2000) 

class Solution:
  
    def minimumPossibleSum(self, n: int, target: int) -> int:
        """
        Calculates the minimum possible sum of a beautiful array of length n with target constraint.

        A beautiful array nums has:
        - nums.length == n
        - nums consists of pairwise distinct positive integers
        - No two distinct indices i, j exist such that nums[i] + nums[j] == target

        Args:
          n: The required length of the array.
          target: The target sum constraint.

        Returns:
          The minimum possible sum modulo 10^9 + 7.
        """

        MOD = 10**9 + 7
        
        # Calculate modular multiplicative inverse of 2 modulo MOD using Fermat's Little Theorem.
        # This is needed for division by 2 in the sum formulas for arithmetic progressions.
        # The formula is a^(p-2) % p = a^-1 % p for prime p. Here p = MOD.
        inv2 = pow(2, MOD - 2, MOD) 

        def sum_1_to_k(k: int) -> int:
            """
            Computes (1 + 2 + ... + k) % MOD using the formula k * (k + 1) / 2.
            Handles large k using modular arithmetic to prevent overflow and stay within MOD.
            """
            # If k is 0 or negative, the sum is 0.
            if k <= 0: 
                return 0
            
            # Calculate k % MOD and (k+1) % MOD.
            # Using k % MOD ensures intermediate products stay manageable,
            # though Python's arbitrary precision integers handle large numbers.
            k_mod = k % MOD
            # Compute (k+1) % MOD efficiently as (k % MOD + 1) % MOD
            k_plus_1_mod = (k_mod + 1) % MOD
            
            # Compute (k * (k + 1)) % MOD
            term1 = (k_mod * k_plus_1_mod) % MOD
            
            # Compute (term1 / 2) % MOD, which is equivalent to (term1 * inv2) % MOD
            result = (term1 * inv2) % MOD
            return result

        def sum_a_to_b(a: int, b: int) -> int:
            """
            Computes (a + (a+1) + ... + b) % MOD using the identity:
            sum(a..b) = sum(1..b) - sum(1..a-1).
            Handles modular subtraction correctly by adding MOD before the final modulo operation.
            """
            # If the range is invalid (a > b), the sum is 0.
            if a > b: 
                return 0
            
            # Calculate sum(1..b) % MOD
            sum_1_b = sum_1_to_k(b)
            
            # Calculate sum(1..a-1) % MOD
            sum_1_a_minus_1 = sum_1_to_k(a - 1)
            
            # Compute (sum(1..b) - sum(1..a-1)) % MOD.
            # Add MOD before taking the final modulo to ensure the result is non-negative.
            result = (sum_1_b - sum_1_a_minus_1 + MOD) % MOD
            return result

        # The greedy strategy to minimize the sum involves picking the smallest available positive integers.
        # An integer `k` cannot be picked if `target - k` has already been picked.
        # Consider the pairs (k, target-k). From each pair with distinct elements, we can pick at most one.
        # To minimize the sum, we prefer the smaller element `k` where `k < target - k`.
        # This applies for k = 1, 2, ..., floor((target-1)/2).
        # The number M = floor(target / 2) represents the count of the smallest integers {1, ..., M}
        # that can be safely included without forming a pair summing to `target` among themselves.
        # Any pair {x, y} from {1, ..., M} has sum x+y <= M + (M-1) = 2M-1.
        # If target = 2M or target = 2M+1, then 2M-1 < target. So {1, ..., M} is fine.
        M = target // 2

        if n <= M:
            # If the required array length `n` is less than or equal to `M`,
            # we can simply pick the first `n` positive integers: {1, 2, ..., n}.
            # As established, no pair among these sums to `target`.
            # This choice gives the minimum possible sum.
            return sum_1_to_k(n)
        else:
            # If n > M, we must include more than M elements.
            # The optimal greedy strategy picks {1, 2, ..., M} first.
            # The sum of these initial M elements is:
            sum1 = sum_1_to_k(M)
            
            # We need to pick an additional (n - M) elements.
            # The greedy strategy would consider M+1, M+2, ...
            # However, any integer `k` such that `target - k` is in {1, ..., M} must be skipped.
            # These skipped integers are exactly {target-M, target-M+1, ..., target-1}.
            # Consequently, the next smallest available integer after M that can be picked is `target`.
            # To get the remaining (n - M) elements while maintaining the minimum sum property,
            # we pick the sequence starting from `target`: {target, target+1, ..., target + (n - M) - 1}.
            
            # Calculate the number of remaining elements needed.
            num_remaining = n - M
            # Determine the range of the second sequence of numbers.
            a = target  # First term of the second sequence
            b = target + num_remaining - 1 # Last term of the second sequence
            
            # Calculate the sum of this second sequence {a, ..., b} using the helper function.
            sum2 = sum_a_to_b(a, b)
            
            # The total minimum sum is the sum of the two parts, modulo MOD.
            total_sum = (sum1 + sum2) % MOD
            return total_sum