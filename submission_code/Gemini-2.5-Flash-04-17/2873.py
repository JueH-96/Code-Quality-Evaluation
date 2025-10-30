from typing import List

class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        # Primes are natural numbers greater than 1. The smallest prime is 2.
        # The sum of two primes must be at least 2 + 2 = 4.
        # Thus, if n < 4, no prime pair [x, y] with x+y=n exists where x, y >= 2.
        # The loop range below correctly handles n=2 and n=3 by being empty.
        # Explicitly checking n < 2 handles the case n=1 where the sieve size might be 0 or cause issues.
        if n < 2:
             return []

        # Use Sieve of Eratosthenes to find all prime numbers up to n efficiently.
        # This creates a boolean list where is_prime[i] is True if i is prime, False otherwise.
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False # 0 and 1 are not prime by definition

        # The Sieve algorithm marks multiples of primes as not prime.
        # We iterate starting from the first prime, 2.
        # We only need to go up to sqrt(n) because any composite number <= n must have a prime factor <= sqrt(n).
        p = 2
        # The loop condition p*p <= n is a standard optimization for the sieve.
        while (p * p <= n):
            # If p is currently marked as prime (i.e., it hasn't been marked as a multiple of a smaller prime)
            if (is_prime[p]):
                # Then p is a prime number. Mark all its multiples as not prime.
                # We start marking from p*p because any smaller multiple (k*p with k < p)
                # would have already been marked as composite by a smaller prime factor of k.
                # We iterate up to n (inclusive).
                for i in range(p * p, n + 1, p):
                    is_prime[i] = False
            p += 1 # Move to the next number to check

        # Now find prime pairs [x, y] such that:
        # 1. x and y are prime numbers
        # 2. x + y = n
        # 3. 1 <= x <= y <= n
        #    The condition x >= 1 is implicitly handled by x being prime, so x>=2.
        #    The condition x <= y and x + y = n implies x <= n - x, which simplifies to 2*x <= n, or x <= n // 2.
        #    The condition y <= n is implied by y = n - x and x >= 2, so y <= n - 2 which is <= n for n>=2.
        # We need to return the pairs sorted by x. By iterating x from smallest to largest, we achieve this automatically.
        prime_pairs = []
        # Iterate through possible values of x. x must be prime, so start from 2.
        # Iterate x up to n // 2 to satisfy the condition x <= y (since y = n - x).
        # The loop range is inclusive of the lower bound (2) and inclusive of the upper bound (n // 2).
        # range(start, stop) includes 'start' but excludes 'stop'. So range(2, n // 2 + 1) is correct.
        for x in range(2, n // 2 + 1):
            # Check if the current value of x is prime using the sieve result.
            if is_prime[x]:
                # If x is prime, the corresponding y must be n - x to satisfy x + y = n.
                y = n - x
                # Check if y is also a prime number.
                # y must be >= 2 to be prime. As discussed, y >= 2 is guaranteed when x is in the loop range [2, n//2] for n >= 4.
                # For n=2,3, the loop is empty. So is_prime[y] check is safe as y >= 2.
                if is_prime[y]:
                    # If both x and y are prime, we have found a valid prime pair [x, y].
                    # Add the pair to the result list. Since we iterate x in increasing order,
                    # the pairs are added in the required sorted order based on x.
                    prime_pairs.append([x, y])

        # Return the list of all found prime pairs, sorted by x.
        return prime_pairs