import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        limit = 32000
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(math.sqrt(limit)) + 1):
            if is_prime[p]:
                for i in range(p * p, limit + 1, p):
                    is_prime[i] = False
        primes = []
        for p in range(2, limit + 1):
            if is_prime[p]:
                primes.append(p)
        
        start_range = max(2, math.ceil(math.sqrt(l)))
        end_range = math.floor(math.sqrt(r))
        
        if start_range > end_range:
            return r - l + 1
            
        special_count = 0
        start_index = -1
        end_index = -1
        
        low = 0
        high = len(primes) - 1
        first_index = -1
        while low <= high:
            mid = (low + high) // 2
            if primes[mid] >= start_range:
                first_index = mid
                high = mid - 1
            else:
                low = mid + 1
                
        last_index = -1
        low = 0
        high = len(primes) - 1
        while low <= high:
            mid = (low + high) // 2
            if primes[mid] <= end_range:
                last_index = mid
                low = mid + 1
            else:
                high = mid - 1
                
        if first_index != -1 and last_index != -1 and first_index <= last_index:
            special_count = last_index - first_index + 1
            
        return r - l + 1 - special_count