import math

class Solution:
    max_prime_limit = 31622  # Since sqrt(10^9) ≈ 31622.776
    primes = []
    pi = []

    def __init__(self):
        self.sieve()
        self.build_pi()

    def sieve(self):
        """Compute all primes up to max_prime_limit using Sieve of Eratosthenes."""
        is_prime = [True] * (self.max_prime_limit + 1)
        p = 2
        while p * p <= self.max_prime_limit:
            if is_prime[p]:
                for i in range(p * p, self.max_prime_limit + 1, p):
                    is_prime[i] = False
            p += 1
        for p in range(2, self.max_prime_limit + 1):
            if is_prime[p]:
                self.primes.append(p)

    def build_pi(self):
        """Build the prime counting function up to max_prime_limit."""
        self.pi = [0] * (self.max_prime_limit + 1)
        for i in range(1, self.max_prime_limit + 1):
            self.pi[i] = self.pi[i - 1] + (1 if i in self.primes else 0)

    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        ceil_sqrt_l = math.isqrt(l - 1) + 1 if l > 0 else 1
        floor_sqrt_r = math.isqrt(r)
        
        if ceil_sqrt_l > floor_sqrt_r:
            return r - l + 1
        
        count_of_squares_of_primes = self.pi[floor_sqrt_r] - self.pi[ceil_sqrt_l - 1]
        return (r - l + 1) - count_of_squares_of_primes

# Example usage:
# sol = Solution()
# print(sol.nonSpecialCount(5, 7))  # Output: 3
# print(sol.nonSpecialCount(4, 16)) # Output: 11