import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number x is special if it has exactly 2 proper divisors.
        # For x > 1, proper divisors are divisors excluding x. Number of proper divisors = d(x) - 1.
        # x is special <=> x > 1 and d(x) - 1 = 2 <=> x > 1 and d(x) = 3.
        # A number x has exactly 3 divisors if and only if it is the square of a prime number (x = p^2 for some prime p).
        # For x = 1, proper divisors are {}. Number of proper divisors = 0. 1 is not special.
        # So, special numbers are squares of prime numbers (p^2).

        # We need to count numbers in the range [l, r] that are not special.
        # This is equivalent to: (Total numbers in [l, r]) - (Count of special numbers in [l, r]).
        # Total numbers in [l, r] is r - l + 1.
        # Special numbers in [l, r] are p^2 such that p is prime and l <= p^2 <= r.
        # l <= p^2 <= r  <=> sqrt(l) <= p <= sqrt(r).
        # Since p must be an integer (and prime), this implies ceil(sqrt(l)) <= p <= floor(sqrt(r)).
        # Also, p must be prime, so the smallest possible value for p is 2.
        # Thus, we need to count primes p such that max(2, ceil(sqrt(l))) <= p <= floor(sqrt(r)).

        # The maximum value of r is 10^9, so the maximum value of floor(sqrt(r)) is floor(sqrt(10^9)) = 31622.
        # We need to identify primes up to this value. We can use a sieve up to a slightly larger limit.
        MAX_P_Sieve = 32000 # A safe upper bound for sqrt(10^9) is 31623, use 32000 for convenience.

        # Sieve of Eratosthenes to find primes up to MAX_P_Sieve.
        isPrime = [True] * (MAX_P_Sieve + 1)
        isPrime[0] = isPrime[1] = False
        # Iterate up to sqrt(MAX_P_Sieve) to mark composites.
        # math.isqrt(32000) is 178.
        for p in range(2, math.isqrt(MAX_P_Sieve) + 1):
            if isPrime[p]:
                # Mark multiples of p starting from p*p as not prime.
                for i in range(p * p, MAX_P_Sieve + 1, p):
                    isPrime[i] = False

        # Calculate the lower bound for prime p: L_final = max(2, ceil(sqrt(l)))
        # ceil(sqrt(l)) can be computed as math.isqrt(l - 1) + 1 for l >= 1.
        ceil_sqrt_l = math.isqrt(l - 1) + 1
        
        # The minimum required prime p is 2. So the effective lower bound for p is max(2, ceil_sqrt_l).
        L_final = max(2, ceil_sqrt_l)

        # Calculate the upper bound for prime p: R_final = floor(sqrt(r)).
        R_final = math.isqrt(r)

        # Count the number of primes p such that L_final <= p <= R_final.
        # These are the bases of the special numbers p^2 that fall within [l, r].
        # Since R_final <= floor(sqrt(10^9)) = 31622 <= MAX_P_Sieve, any prime p in the range [L_final, R_final]
        # will have p <= R_final <= MAX_P_Sieve, so we can safely query the isPrime array.
        special_count = 0
        
        # Iterate through the possible prime bases p in the valid range [L_final, R_final].
        # The range function handles the case where L_final > R_final by producing an empty sequence.
        for p in range(L_final, R_final + 1):
            # Check if the current number p is marked as prime in our sieve.
            if isPrime[p]:
                special_count += 1

        # The total count of numbers in the range [l, r] is r - l + 1.
        total_numbers = r - l + 1

        # The count of non-special numbers is the total numbers minus the count of special numbers found.
        return total_numbers - special_count