class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        from math import gcd
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def lcm_of_list(lst):
            result = lst[0]
            for x in lst[1:]:
                result = lcm(result, x)
            return result
        
        def cost_to_make_multiple(n, t):
            remainder = n % t
            return 0 if remainder == 0 else t - remainder
        
        n_targets = len(target)
        n_subsets = 1 << n_targets
        
        # For each non-empty subset of targets, compute the minimum cost to satisfy all of them
        # by making at least one element in nums a multiple of their LCM
        min_cost_for_subset = [float('inf')] * n_subsets
        min_cost_for_subset[0] = 0
        
        for mask in range(1, n_subsets):
            subset = [target[i] for i in range(n_targets) if mask & (1 << i)]
            lcm_val = lcm_of_list(subset)
            min_cost_for_subset[mask] = min(cost_to_make_multiple(n, lcm_val) for n in nums)
        
        # dp[mask] = minimum cost to satisfy all targets in mask
        dp = [float('inf')] * n_subsets
        dp[0] = 0
        
        for mask in range(1, n_subsets):
            # Try all non-empty submasks of mask
            submask = mask
            while submask > 0:
                dp[mask] = min(dp[mask], dp[mask ^ submask] + min_cost_for_subset[submask])
                submask = (submask - 1) & mask
        
        return dp[n_subsets - 1]