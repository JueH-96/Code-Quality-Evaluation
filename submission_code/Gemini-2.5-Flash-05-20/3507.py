import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number is "special" if it has exactly 2 proper divisors.
        # This property is exclusive to squares of prime numbers.
        # For a prime p, the divisors of p*p are 1, p, and p*p.
        # The proper divisors (divisors excluding p*p itself) are 1 and p, which is exactly 2.

        # The problem asks for the count of numbers in the range [l, r] that are NOT special.
        # This can be calculated as:
        # (Total numbers in range [l, r]) - (Count of special numbers in range [l, r])

        # Step 1: Determine the maximum value for a prime 'p' we need to consider.
        # A special number x = p*p must be in the range [l, r].
        # So, l <= p*p <= r, which implies sqrt(l) <= p <= sqrt(r).
        # We need to find all prime numbers up to floor(sqrt(r)).
        max_p_to_sieve = int(math.sqrt(r))

        # Step 2: Use the Sieve of Eratosthenes to find all prime numbers up to max_p_to_sieve.
        # `is_prime[i]` will be True if 'i' is a prime number, False otherwise.
        is_prime = [True] * (max_p_to_sieve + 1)

        # 0 and 1 are not prime numbers.
        if max_p_to_sieve >= 0:
            is_prime[0] = False
        if max_p_to_sieve >= 1:
            is_prime[1] = False

        # Iterate from 2 up to sqrt(max_p_to_sieve).
        # For each prime 'p', mark its multiples as non-prime.
        for p in range(2, int(math.sqrt(max_p_to_sieve)) + 1):
            if is_prime[p]:
                # Mark multiples of p as not prime, starting from p*p.
                # Smaller multiples (e.g., 2*p, 3*p) would have been marked by smaller prime factors.
                for multiple in range(p * p, max_p_to_sieve + 1, p):
                    is_prime[multiple] = False

        # Step 3: Count how many special numbers (squares of primes) fall within the range [l, r].
        special_count = 0
        # Iterate through numbers from 2 up to max_p_to_sieve.
        for p in range(2, max_p_to_sieve + 1):
            if is_prime[p]:
                p_squared = p * p
                # Check if this prime square is within the specified range [l, r].
                # Note: Python's integers handle arbitrary size, so p*p will not overflow.
                # The maximum p_squared will be (31622)^2 approx 10^9, fitting in standard 64-bit int.
                if l <= p_squared <= r:
                    special_count += 1
        
        # Step 4: Calculate the total number of integers in the range [l, r].
        total_numbers_in_range = r - l + 1
        
        # Step 5: Subtract the count of special numbers from the total count to get the non-special count.
        return total_numbers_in_range - special_count