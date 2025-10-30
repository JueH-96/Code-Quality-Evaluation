class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        import math
        
        # Find all primes up to sqrt(r)
        max_prime = int(math.isqrt(r)) + 1
        is_prime = [True] * (max_prime)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(max_prime)) + 1):
            if is_prime[i]:
                for j in range(i*i, max_prime, i):
                    is_prime[j] = False
        primes = [i for i, prime in enumerate(is_prime) if prime]
        
        # Count the number of special numbers in [l, r]
        special_count = 0
        for p in primes:
            p_squared = p * p
            if l <= p_squared <= r:
                special_count += 1
            elif p_squared > r:
                break  # No need to check further as p^2 will only increase
        
        total_numbers = r - l + 1
        non_special_count = total_numbers - special_count
        return non_special_count