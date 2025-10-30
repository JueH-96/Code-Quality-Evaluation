import math

class Solution:
  def nonSpecialCount(self, l: int, r: int) -> int:
    """
    Counts the numbers in the range [l, r] that are not special.
    A number is special if it has exactly 2 proper divisors.
    This implies special numbers are of the form p^2, where p is a prime.
    """

    # Total numbers in the range [l, r] inclusive.
    total_numbers_in_range = r - l + 1

    # We need to find special numbers (p^2) such that l <= p^2 <= r.
    # This is equivalent to finding primes p such that sqrt(l) <= p <= sqrt(r).
    
    # The maximum possible value for a prime p such that p^2 <= r is floor(sqrt(r)).
    # Let this be max_p_limit.
    # Smallest prime is 2. If max_p_limit < 2, there are no relevant primes.
    max_p_limit = int(math.sqrt(r))

    special_count = 0
    
    if max_p_limit >= 2:
        # Sieve of Eratosthenes to find all primes up to max_p_limit.
        # is_prime[i] will be True if i is prime, False otherwise.
        # Array size is max_p_limit + 1 to include index max_p_limit.
        is_prime = [True] * (max_p_limit + 1)
        is_prime[0] = False  # 0 is not prime
        is_prime[1] = False  # 1 is not prime
        
        # Optimized Sieve: The outer loop for 'num' only needs to go up to sqrt(max_p_limit).
        sieve_loop_upper_bound = int(math.sqrt(max_p_limit))
        for num in range(2, sieve_loop_upper_bound + 1):
            if is_prime[num]:
                # If num is prime, mark all its multiples (starting from num*num) as not prime.
                for multiple in range(num * num, max_p_limit + 1, num):
                    is_prime[multiple] = False
        
        # Count the special numbers (p^2) that fall within the range [l, r].
        # Iterate through all numbers p from 2 up to max_p_limit.
        for p in range(2, max_p_limit + 1):
            if is_prime[p]:  # If p is a prime number
                p_squared = p * p
                
                # A special number p^2 must be within the given range [l, r].
                # We know p <= max_p_limit = floor(sqrt(r)).
                # Since p is an integer, p <= sqrt(r).
                # Squaring both sides (non-negative values): p^2 <= r.
                # So, the condition p_squared <= r is implicitly satisfied by iterating p up to max_p_limit.
                # We only need to explicitly check if p_squared >= l.
                if p_squared >= l:
                    special_count += 1
                    
    return total_numbers_in_range - special_count