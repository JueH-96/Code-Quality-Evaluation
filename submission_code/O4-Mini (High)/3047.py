class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        import math
        from collections import defaultdict
        
        # Compute all primes up to sqrt(max_num) via sieve
        max_num = max(nums)
        maxp = math.isqrt(max_num)
        sieve = [True] * (maxp + 1)
        sieve[0:2] = [False, False]
        limit = math.isqrt(maxp)
        for i in range(2, limit + 1):
            if sieve[i]:
                for j in range(i * i, maxp + 1, i):
                    sieve[j] = False
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        
        # Group numbers by their square-free part and sum
        sum_by_sf = defaultdict(int)
        ans = 0
        for x in nums:
            y = x
            # Remove all square factors p^2 from y
            for p in primes:
                pp = p * p
                if pp > y:
                    break
                while y % pp == 0:
                    y //= pp
            # y is now the square-free part
            sum_by_sf[y] += x
            if sum_by_sf[y] > ans:
                ans = sum_by_sf[y]
        
        return ans