class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def lcm_of_list(lst):
            return reduce(lcm, lst)
        
        m = len(target)
        
        # DP: dp[mask] = minimum cost to satisfy targets in mask
        dp = [float('inf')] * (1 << m)
        dp[0] = 0
        
        for num in nums:
            new_dp = dp[:]
            for mask in range(1, 1 << m):
                subset = [target[j] for j in range(m) if mask & (1 << j)]
                lcm_val = lcm_of_list(subset)
                cost = ((num + lcm_val - 1) // lcm_val) * lcm_val - num
                
                for prev_mask in range(1 << m):
                    if dp[prev_mask] == float('inf'):
                        continue
                    new_mask = prev_mask | mask
                    new_dp[new_mask] = min(new_dp[new_mask], dp[prev_mask] + cost)
            
            dp = new_dp
        
        return dp[(1 << m) - 1]