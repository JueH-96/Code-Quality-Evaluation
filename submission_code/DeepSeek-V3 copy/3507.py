import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        def count_special(l, r):
            count = 0
            # Special numbers are squares of primes
            # So we need to find all primes p such that p^2 is in [l, r]
            # The maximum p is sqrt(r)
            max_p = int(math.sqrt(r)) + 1
            for p in range(2, max_p):
                if is_prime(p):
                    square = p * p
                    if l <= square <= r:
                        count += 1
            return count

        special_count = count_special(l, r)
        total_numbers = r - l + 1
        return total_numbers - special_count