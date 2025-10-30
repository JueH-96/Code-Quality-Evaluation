import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # Function to count primes in [a, b] using Sieve of Eratosthenes.
        def count_primes_in_range(a, b):
            if b < 2:
                return 0
            sieve_bound = b + 1
            sieve = [True] * sieve_bound
            sieve[0] = sieve[1] = False
            for i in range(2, int(b ** 0.5) + 1):
                if sieve[i]:
                    for j in range(i * i, sieve_bound, i):
                        sieve[j] = False
            count = 0
            for num in range(a, b+1):
                if num < len(sieve) and sieve[num]:
                    count += 1
            return count
        
        # Special numbers are those with exactly 2 proper divisors.
        # Such numbers have exactly 3 total divisors.
        # It is known that numbers with exactly 3 divisors are exactly the squares of primes.
        # So we need to count prime squares in the range [l, r].
        # We find the range of primes p such that p^2 is in [l, r].
        start = int(math.ceil(math.sqrt(l)))
        end = int(math.floor(math.sqrt(r)))
        
        special_count = 0
        if start <= end:
            special_count = count_primes_in_range(start, end)
        
        total_count = r - l + 1
        
        return total_count - special_count

# Example usage:
# sol = Solution()
# print(sol.nonSpecialCount(5, 7))  # Expected output: 3
# print(sol.nonSpecialCount(4, 16)) # Expected output: 11