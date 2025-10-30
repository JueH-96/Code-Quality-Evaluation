from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        """
        Finds pairs of prime numbers [x, y] such that x + y == n, 1 <= x <= y <= n.
        The result is a 2D sorted list, sorted in increasing order of x.

        Args:
            n: An integer.

        Returns:
            A list of prime number pairs.
        """

        # Handle base cases where no prime pairs are possible.
        # The smallest sum of two prime numbers is 2 + 2 = 4.
        # So, if n is less than 4, no such pairs can exist.
        if n < 4:
            return []

        # Step 1: Sieve of Eratosthenes to efficiently identify all prime numbers up to n.
        # Create a boolean list `is_prime` where `is_prime[i]` is True if i is prime, False otherwise.
        is_prime = [True] * (n + 1)
        is_prime[0] = False  # 0 is not prime
        is_prime[1] = False  # 1 is not prime

        # Iterate from 2 up to sqrt(n).
        # Any composite number `k` less than or equal to `n` must have at least one prime factor
        # less than or equal to sqrt(k). Thus, to mark all composites up to `n`,
        # we only need to iterate prime `p` up to sqrt(n).
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                # Mark all multiples of p (starting from p*p) as not prime.
                # Multiples smaller than p*p (e.g., 2p, 3p, etc.) would have already been marked
                # by smaller prime factors.
                for multiple in range(p * p, n + 1, p):
                    is_prime[multiple] = False

        # Step 2: Iterate through possible values of x and find corresponding y.
        # x must be a prime number (i.e., is_prime[x] must be True).
        # The smallest prime is 2, so x starts from 2.
        # The condition 1 <= x <= y <= n implies:
        #   x >= 1 (covered by x >= 2)
        #   x <= y (since x + y = n, this means x <= n - x => 2x <= n => x <= n/2)
        #   y <= n (since x >= 2, y = n - x <= n - 2, so y <= n is guaranteed)
        
        result_pairs = []
        # Iterate x from 2 up to n // 2 (inclusive)
        for x in range(2, n // 2 + 1):
            # Check if x is prime.
            if is_prime[x]:
                y = n - x
                # Check if y is prime.
                # y is guaranteed to be >= x due to the loop condition x <= n // 2.
                if is_prime[y]:
                    result_pairs.append([x, y])
        
        # The list `result_pairs` is naturally sorted by x because we iterate x in increasing order
        # and append pairs as they are found.
        return result_pairs