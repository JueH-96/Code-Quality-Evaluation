class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number is special if it has exactly 2 proper divisors,
        # i.e., it has exactly 3 divisors (1, a, x where a^2 = x and a is prime).
        # So the special numbers are precisely the perfect squares of prime numbers.
        # Our task is then to count numbers in [l, r] that are NOT special.
        # Total count in range = (r - l + 1)
        # Let count_special be number of prime squares in [l, r].
        # Then answer = (r - l + 1) - count_special
        
        # To find all primes p such that p^2 is between l and r,
        # p should be between sqrt(l) and sqrt(r). But since l might not be a perfect square,
        # we iterate p from 2 to floor(sqrt(r)) and check if p is prime,
        # and then if p**2 is in the range.
        
        import math
        
        # Upper bound for possible prime numbers whose square is <= r.
        upper = int(math.isqrt(r))
        
        # Sieve of Eratosthenes to get all primes up to upper.
        sieve = [True] * (upper + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(upper)) + 1):
            if sieve[i]:
                for j in range(i*i, upper+1, i):
                    sieve[j] = False
        
        # Count primes p for which p^2 is in [l, r].
        count_special = 0
        for p in range(2, upper + 1):
            if sieve[p]:
                square = p * p
                if l <= square <= r:  # Check if prime square is within the range.
                    count_special += 1
        
        total_numbers = r - l + 1
        return total_numbers - count_special