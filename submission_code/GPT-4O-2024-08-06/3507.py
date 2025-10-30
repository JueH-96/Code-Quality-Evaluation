class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        # A number is special if it is a perfect square of a prime number.
        # We need to find all such numbers in the range [l, r].
        
        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        
        # Find all perfect squares of primes within the range [l, r]
        special_count = 0
        p = 2
        while p * p <= r:
            if is_prime(p):
                square = p * p
                if l <= square <= r:
                    special_count += 1
            p += 1
        
        # Total numbers in the range [l, r]
        total_count = r - l + 1
        
        # Non-special numbers are total numbers minus special numbers
        return total_count - special_count