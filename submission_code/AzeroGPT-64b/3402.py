class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        from collections import Counter
        from functools import lru_cache
        mod = 10**9 + 7
        nums_map = Counter(nums)
        
        sorted_keys = sorted([k for k in nums_map])
        n = len(nums)
        pref_sum = [0] * (n + 1)
        
        @lru_cache(None)
        def go_to(ith, step_cost):
            if ith == n:
                return 0
            val = sorted_keys[ith]
            res = 0
            for next_ith in range(ith, n):
                next_val = sorted_keys[next_ith]
                res += nums_map[next_val] * ((next_val - val) * (next_ith - ith + 1 - pref_sum[ith]) + go_to(next_ith + 1, cost2 if next_val != val else cost1))
            return res % mod
        
        if cost1 <= cost2:
            return (sum([(k - min(nums)) * v for k, v in nums_map.items()]) * cost1) % mod
        
        steps = 0
        for k in nums_map.keys():
            if k < sorted_keys[0]:
                steps += nums_map[k]
            elif k > sorted_keys[-1]:
                steps += nums_map[k]
        pref_sum[steps] = steps
            
        for i in range(steps - 1, -1, -1):
            pref_sum[i] = pref_sum[i + 1] + 1
        
        value = go_to(0, cost1)
        return value