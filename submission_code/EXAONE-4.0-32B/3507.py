import math
import bisect

class Solution:
    _primes_list = None

    def nonSpecialCount(self, l: int, r: int) -> int:
        if Solution._primes_list is None:
            max_n = 31622
            sieve = [True] * (max_n + 1)
            sieve[0] = False
            sieve[1] = False
            i = 2
            while i * i <= max_n:
                if sieve[i]:
                    j = i * i
                    while j <= max_n:
                        sieve[j] = False
                        j += i
                i += 1
            Solution._primes_list = [num for num in range(2, max_n + 1) if sieve[num]]
        
        total = r - l + 1
        start_val = math.isqrt(l)
        if start_val * start_val < l:
            start_val += 1
        end_val = math.isqrt(r)
        
        if start_val > end_val:
            return total
        
        left_idx = bisect.bisect_left(Solution._primes_list, start_val)
        right_idx = bisect.bisect_right(Solution._primes_list, end_val)
        count_special = right_idx - left_idx
        
        return total - count_special