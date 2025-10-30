import math
from collections import defaultdict
from typing import List

class Solution:
    # Precompute primes up to sqrt(10^9) + epsilon.
    # The maximum value in nums is 10^9, so its square root is approximately 31622.
    # We need primes up to this value for efficient factorization.
    # Adding +2 ensures the upper bound for the sieve is sufficient (e.g., handles int truncation).
    MAX_SQRT_NUM = int(math.sqrt(10**9)) + 2

    # Class-level variables to store precomputed primes and the sieve array.
    # This ensures computation happens only once across multiple test cases if run in the same session.
    _primes = []
    _is_prime_sieve = []

    # Static block for precomputation using Sieve of Eratosthenes.
    # This code runs when the class is first loaded.
    if not _primes: # Check if precomputation has already occurred to avoid re-running.
        _is_prime_sieve = [True] * MAX_SQRT_NUM
        _is_prime_sieve[0] = _is_prime_sieve[1] = False # 0 and 1 are not prime numbers.
        
        for p in range(2, MAX_SQRT_NUM):
            if _is_prime_sieve[p]:
                _primes.append(p)
                # Mark multiples of p as not prime, starting from p*p for efficiency
                # (multiples less than p*p would have been marked by smaller prime factors).
                for multiple in range(p*p, MAX_SQRT_NUM, p):
                    _is_prime_sieve[multiple] = False

    def _get_square_free_part(self, n: int) -> int:
        """
        Calculates the square-free part of a given integer n.
        The square-free part of n is the product of all prime factors of n
        that appear with an odd exponent in its prime factorization.
        Example:
        - For 12 (2^2 * 3^1), square-free part is 3.
        - For 75 (3^1 * 5^2), square-free part is 3.
        - For 8 (2^3), square-free part is 2.
        
        The precomputed `_primes` list is used for efficient factorization.
        """
        sf_part = 1
        temp_n = n # Use a temporary variable as we will modify it during factorization.
        
        for p in self._primes:
            # Optimization: If p*p is greater than temp_n, it means any remaining
            # factor of temp_n must be a prime number itself (or 1),
            # or a product of such primes. Since we have checked all primes up to
            # sqrt(original_n), any remaining temp_n > 1 must be a prime factor
            # whose value is greater than MAX_SQRT_NUM.
            if p * p > temp_n:
                break
            
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                # If the prime factor p appears an odd number of times,
                # it contributes to the square-free part.
                if count % 2 == 1:
                    sf_part *= p
        
        # After iterating through all precomputed primes, if `temp_n` is still
        # greater than 1, it means the remaining `temp_n` is a prime factor
        # (whose value is greater than MAX_SQRT_NUM). This prime factor also
        # contributes to the square-free part.
        if temp_n > 1:
            sf_part *= temp_n
            
        return sf_part

    def maximumSum(self, nums: List[int]) -> int:
        """
        Finds the maximum element-sum of a complete subset of indices.
        A set of numbers is complete if the product of every pair of its elements
        is a perfect square. This condition is mathematically equivalent to
        all numbers in the set having the same square-free part.
        """
        
        # defaultdict to store the sum of numbers for each unique square-free part.
        # Keys: square-free parts, Values: sum of numbers associated with that square-free part.
        sf_sums = defaultdict(int)
        
        # Iterate through each number in the input list.
        for num_val in nums:
            # Calculate the square-free part for the current number.
            sfp = self._get_square_free_part(num_val)
            # Add the number's value to the sum corresponding to its square-free part.
            sf_sums[sfp] += num_val
            
        # The maximum element-sum of a complete subset is the maximum value
        # among all the accumulated sums for different square-free parts.
        # Every single element forms a complete subset, so sf_sums will not be empty
        # (given nums has at least one element by constraints), and `max` will work.
        return max(sf_sums.values())