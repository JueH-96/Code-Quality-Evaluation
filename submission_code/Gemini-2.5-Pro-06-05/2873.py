from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        """
        Finds all prime number pairs [x, y] that sum to n.
        """
        
        # Step 1: Sieve of Eratosthenes to find all prime numbers up to n.
        # This pre-computation allows for O(1) primality tests later.
        is_prime = [True] * (n + 1)
        
        # 0 and 1 are not prime numbers. Given the constraint 1 <= n,
        # the list size is at least 2, so these indices are safe to access.
        if n >= 1:
            is_prime[0] = is_prime[1] = False
        elif n == 0:
            is_prime[0] = False

        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                # If p is prime, mark all its multiples as not prime.
                # We can start marking from p*p, as any smaller multiple
                # of p would have a prime factor smaller than p, and would
                # have been marked already.
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
        
        # Step 2: Find the prime pairs.
        result = []
        
        # Iterate through possible values for the first number, x.
        # The constraint x <= y, combined with x + y = n, implies x <= n - x,
        # which simplifies to 2x <= n or x <= n/2.
        # We start from 2, the smallest prime.
        for x in range(2, n // 2 + 1):
            # If x is a prime number...
            if is_prime[x]:
                y = n - x
                # ...and its complement y is also a prime number...
                if is_prime[y]:
                    # ...then we have found a valid pair.
                    result.append([x, y])
                    
        return result