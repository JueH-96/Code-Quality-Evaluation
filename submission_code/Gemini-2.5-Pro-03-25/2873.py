import math
from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        """
        Finds all prime number pairs [x, y] such that x + y = n, 1 <= x <= y <= n,
        and both x and y are prime numbers.

        Args:
            n: The target sum for the prime pairs. Constraints: 1 <= n <= 10^6.

        Returns:
            A 2D sorted list of prime number pairs [x_i, y_i] satisfying the conditions.
            The list is sorted in increasing order of x_i.
            Returns an empty list if no such pairs exist.
        
        Example:
            findPrimePairs(10) should return [[3, 7], [5, 5]]
            findPrimePairs(2) should return []
        """

        # Step 1: Sieve of Eratosthenes to find all prime numbers up to n.
        # Create a boolean array `is_prime` of size n+1, initialized to True.
        # is_prime[i] will be True if i is prime, and False otherwise.
        is_prime = [True] * (n + 1)

        # 0 and 1 are not prime numbers.
        # Mark them as False. Note: n is guaranteed to be >= 1 by constraints.
        if n >= 0: # Check added for robustness, although n >= 1 is guaranteed.
             is_prime[0] = False
        if n >= 1:
             is_prime[1] = False

        # Implement the Sieve algorithm.
        # Iterate from p = 2 up to the square root of n.
        for p in range(2, int(math.sqrt(n)) + 1):
            # If p is still marked as prime (it hasn't been crossed out by a smaller prime factor)
            if is_prime[p]:
                # Mark all multiples of p as not prime.
                # We can start marking from p*p because any smaller multiple of p (like k*p where k < p)
                # would have already been marked by the prime factor k.
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False

        # Step 2: Find all pairs [x, y] that satisfy the conditions.
        result = []
        
        # Iterate through possible values for x, the first number in the pair.
        # The conditions are:
        # 1. 1 <= x <= y <= n
        # 2. x + y == n
        # 3. x and y are prime numbers.
        
        # From condition 3, x must be prime, so x >= 2.
        # From conditions 1 and 2 (x <= y and x + y = n), we deduce 2x <= x + y = n, so x <= n / 2.
        # Therefore, we only need to check potential values of x in the range [2, n // 2].
        for x in range(2, n // 2 + 1):
            # Check if the current number x is prime using our precomputed sieve results.
            if is_prime[x]:
                # Calculate the potential partner y such that x + y = n.
                y = n - x
                
                # Check if y is also prime.
                # Since x >= 2, y = n - x <= n - 2 (for n >= 4). y is always <= n.
                # Since x <= n/2, y = n - x >= n - n/2 = n/2. Thus, y >= x is guaranteed.
                # The index y will be within the valid range [0, n] of the is_prime array.
                if is_prime[y]:
                    # If both x and y are prime, and they sum to n, and x <= y,
                    # then we have found a valid prime pair. Add it to the result list.
                    result.append([x, y])

        # The loop iterates through x in increasing order (from 2 up to n // 2).
        # Consequently, the pairs [x, y] are added to the `result` list
        # sorted by the first element x. No additional sorting is needed.
        return result