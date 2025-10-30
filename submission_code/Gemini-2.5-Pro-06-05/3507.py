import math

class Solution:
  def nonSpecialCount(self, l: int, r: int) -> int:
    # A number is special if it has exactly 2 proper divisors.
    # This means the number has a total of 3 divisors.
    # A number has 3 divisors if and only if it is the square of a prime number (p^2).
    
    # We need to find the count of numbers in [l, r] that are NOT p^2.
    # This is (total numbers in [l, r]) - (count of special numbers in [l, r]).

    total_count = r - l + 1

    # We need to count primes p such that l <= p^2 <= r.
    # This is equivalent to finding primes p such that sqrt(l) <= p <= sqrt(r).

    # The maximum value of r is 10^9, so sqrt(r) is at most ~31622.
    # We can use a Sieve of Eratosthenes to find all primes up to sqrt(r).
    limit = int(math.sqrt(r))
    
    # The smallest special number is 4 (2^2). If r < 4, no special numbers exist.
    if limit < 2:
      return total_count
      
    # is_prime[i] will be true if i is a prime number.
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(math.sqrt(limit)) + 1):
      if is_prime[p]:
        # Mark all multiples of p as not prime, starting from p*p.
        for i in range(p * p, limit + 1, p):
          is_prime[i] = False

    # Now, count the special numbers. These are p^2 where p is prime and l <= p^2 <= r.
    # This implies that the prime p must be in the range [ceil(sqrt(l)), floor(sqrt(r))].
    special_count = 0
    start_p = int(math.ceil(math.sqrt(l)))
    
    # Iterate through the possible prime bases from ceil(sqrt(l)) to floor(sqrt(r))
    # and check if they are prime using our pre-computed sieve.
    for p in range(start_p, limit + 1):
      if is_prime[p]:
        special_count += 1
    
    return total_count - special_count