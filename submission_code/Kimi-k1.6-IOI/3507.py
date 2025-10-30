import math
import bisect

class Solution:
    _primes = None  # Class variable to store primes
    
    @classmethod
    def _generate_primes(cls):
        if cls._primes is None:
            max_prime = 31622  # sqrt(1e9) is approximately 31622.776
            is_prime = [True] * (max_prime + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(math.sqrt(max_prime)) + 1):
                if is_prime[i]:
                    for j in range(i * i, max_prime + 1, i):
                        is_prime[j] = False
            cls._primes = [i for i, val in enumerate(is_prime) if val]
    
    def nonSpecialCount(self, l: int, r: int) -> int:
        self._generate_primes()
        primes_list = self._primes
        
        # Calculate the lower and upper bounds for primes p where p^2 is in [l, r]
        lower = math.ceil(math.sqrt(l))
        upper = math.floor(math.sqrt(r))
        lower = max(2, lower)  # Primes start from 2
        
        special_count = 0
        if lower <= upper:
            # Find the number of primes between lower and upper inclusive
            left = bisect.bisect_left(primes_list, lower)
            right = bisect.bisect_right(primes_list, upper)
            special_count = right - left
        
        total_numbers = r - l + 1
        return total_numbers - special_count