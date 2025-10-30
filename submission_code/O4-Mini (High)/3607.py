from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        import math
        
        n = len(nums)
        if n <= 1:
            return 0
        
        # Build list of primes up to sqrt(max(nums))
        max_val = max(nums)
        limit = int(math.isqrt(max_val)) + 1
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        
        # Function to get smallest prime factor (or 1 for x <= 1)
        def get_spf(x: int) -> int:
            if x <= 1:
                return 1
            for p in primes:
                if p * p > x:
                    break
                if x % p == 0:
                    return p
            return x
        
        INF = float('inf')
        
        # Initialize DP for the first element
        v0 = nums[0]
        spf0 = get_spf(v0)
        if spf0 != v0:
            dp0_cost, dp1_cost = 0, 1
        else:
            dp0_cost, dp1_cost = 0, INF
        dp0_last, dp1_last = v0, spf0
        
        # Process rest of the array
        for x in nums[1:]:
            v0_i = x
            v1_i = get_spf(x)
            
            new0 = INF
            new1 = INF
            
            # Transition from state 0 (we used v0_last)
            if dp0_cost < INF:
                if dp0_last <= v0_i:
                    new0 = dp0_cost
                if v1_i != v0_i and dp0_last <= v1_i:
                    new1 = dp0_cost + 1
            
            # Transition from state 1 (we used v1_last)
            if dp1_cost < INF:
                if dp1_last <= v0_i and dp1_cost < new0:
                    new0 = dp1_cost
                if v1_i != v0_i and dp1_last <= v1_i and dp1_cost + 1 < new1:
                    new1 = dp1_cost + 1
            
            # If both states are impossible, answer is -1
            if new0 == INF and new1 == INF:
                return -1
            
            dp0_cost, dp1_cost = new0, new1
            dp0_last, dp1_last = v0_i, v1_i
        
        return min(dp0_cost, dp1_cost)