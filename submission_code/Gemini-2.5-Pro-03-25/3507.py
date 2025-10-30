import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        """
        Calculates the count of non-special numbers in the range [l, r].
        A number is defined as special if it has exactly two proper divisors.
        This condition holds if and only if the number is the square of a prime number (p^2).
        The function counts the numbers in the given range [l, r] that are NOT squares of prime numbers.

        Args:
          l: The lower bound of the range (inclusive). 1 <= l <= r.
          r: The upper bound of the range (inclusive). l <= r <= 10^9.

        Returns:
          The count of non-special numbers in the range [l, r].
        """
        
        # The core idea is to count the number of special numbers in the range [l, r]
        # and subtract this count from the total number of integers in the range.
        # A number x is special if x = p^2 for some prime p.
        # We need to find the count of such x where l <= x <= r.
        # This is equivalent to finding primes p such that l <= p^2 <= r.
        # Taking the square root, we get sqrt(l) <= p <= sqrt(r).
        # So, we need to count the number of primes p in the interval [ceil(sqrt(l)), floor(sqrt(r))].

        # Calculate the upper bound for the prime search: floor(sqrt(r)).
        max_p = int(math.sqrt(r))
        
        # Calculate the lower bound for the prime search: ceil(sqrt(l)).
        # We use integer arithmetic to avoid potential floating-point precision issues.
        # Find the smallest integer p such that p * p >= l.
        sqrt_l_floor = int(math.sqrt(l))
        if sqrt_l_floor * sqrt_l_floor < l:
            # If floor(sqrt(l))^2 < l, then ceil(sqrt(l)) = floor(sqrt(l)) + 1.
            min_p_int = sqrt_l_floor + 1
        else:
            # If floor(sqrt(l))^2 >= l, then ceil(sqrt(l)) = floor(sqrt(l)). 
            # This also handles the case where l is a perfect square.
            min_p_int = sqrt_l_floor

        # The maximum value up to which we need to run the sieve is max_p.
        limit = max_p
        special_count = 0 # Initialize count of special numbers in the range.

        # We only need to perform the sieve and count primes if the limit is at least 2,
        # as 2 is the smallest prime number.
        if limit >= 2:
            # Use the Sieve of Eratosthenes to find all prime numbers up to 'limit'.
            is_prime = [True] * (limit + 1) # Boolean array to mark primality.
            
            # 0 and 1 are not prime numbers. Mark them as False.
            # Need to check if indices exist before assignment, although limit >= 2 ensures this.
            is_prime[0] = False
            is_prime[1] = False
                
            # Perform the sieve algorithm.
            # Iterate from 2 up to sqrt(limit).
            for p in range(2, int(math.sqrt(limit)) + 1):
                # If p is prime (not marked False yet):
                if is_prime[p]:
                    # Mark all multiples of p (starting from p*p) as not prime.
                    # Multiples less than p*p would have already been marked by smaller primes.
                    for i in range(p * p, limit + 1, p):
                        is_prime[i] = False
            
            # Now count the primes p that fall within our required range [min_p_int, max_p].
            # Primes must be at least 2. So, the effective start of our check is max(2, min_p_int).
            start_prime_check = max(2, min_p_int)
            
            # Iterate through the numbers from start_prime_check up to limit (max_p).
            for p in range(start_prime_check, limit + 1):
                # If the number p is marked as prime by the sieve:
                if is_prime[p]:
                    # Then p is a prime number such that ceil(sqrt(l)) <= p <= floor(sqrt(r)).
                    # This means p*p is a special number within the range [l, r].
                    special_count += 1
                        
        # Calculate the total number of integers in the range [l, r].
        total_count = r - l + 1
        
        # The count of non-special numbers is the total count minus the count of special numbers.
        return total_count - special_count