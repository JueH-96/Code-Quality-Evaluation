from typing import List

class Solution:
  def findPrimePairs(self, n: int) -> List[List[int]]:
    # Sieve of Eratosthenes to find all primes up to n
    # is_prime[i] is True if i is prime, False otherwise.
    # Size n+1 to cover numbers from 0 to n.
    is_prime = [True] * (n + 1)

    # 0 and 1 are not prime.
    # Given constraint n >= 1, n + 1 >= 2.
    # So, is_prime list has at least 2 elements (indices 0 and 1).
    # Accessing is_prime[0] and is_prime[1] is safe.
    is_prime[0] = False 
    is_prime[1] = False

    # Sieve algorithm
    # Iterate from p = 2 up to sqrt(n).
    # int(n**0.5) is equivalent to int(math.sqrt(n)).
    # The upper limit for p in range should be int(n**0.5) + 1 because range is exclusive at the end.
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            # Mark all multiples of p as not prime.
            # Start marking from p*p, as smaller multiples (e.g., 2*p, 3*p)
            # would have already been marked by smaller primes (2, 3, ...).
            # The multiples are p*p, p*p + p, p*p + 2*p, ..., up to n.
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
    
    result = []
    
    # Find pairs [x, y] satisfying the conditions:
    # 1. 1 <= x <= y <= n
    # 2. x + y == n
    # 3. x and y are prime numbers
    
    # Iterate x from 2 (smallest prime) up to n/2.
    # The condition x <= y implies x <= n - x, so 2x <= n, or x <= n/2.
    # Iterating x up to n // 2 (integer division) ensures x <= y.
    # The loop for x is range(2, n // 2 + 1) because range's second argument is exclusive.
    for x in range(2, n // 2 + 1):
        if is_prime[x]:  # Check if x is prime
            y = n - x
            # y must also be prime.
            # Since x >= 2 (smallest prime), y = n - x <= n - 2.
            # This means y is always within the bounds of the is_prime array [0, n].
            # So, is_prime[y] check is safe.
            if is_prime[y]:  # Check if y is prime
                result.append([x, y])
                
    # The pairs are added to 'result' in increasing order of x,
    # because the loop for x iterates from 2 upwards.
    # Thus, 'result' is already sorted as required, and no extra sort is needed.
    return result