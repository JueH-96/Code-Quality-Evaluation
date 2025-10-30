class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        from functools import lru_cache
        
        # For each target, calculate cost to convert each num to its multiple
        def get_cost(num, tgt):
            # Find the smallest multiple of tgt that is >= num
            multiple = ((num + tgt - 1) // tgt) * tgt
            return multiple - num
        
        # Create cost matrix: cost[i][j] = cost to make nums[i] a multiple of target[j]
        costs = []
        for num in nums:
            row = []
            for tgt in target:
                row.append(get_cost(num, tgt))
            costs.append(row)
        
        n = len(nums)
        m = len(target)
        
        @lru_cache(None)
        def dp(target_mask, used_nums):
            # target_mask: bitmask of which targets are already satisfied
            # used_nums: bitmask of which nums are already used
            
            if target_mask == (1 << m) - 1:  # All targets satisfied
                return 0
            
            # Find next unsatisfied target
            next_target = -1
            for i in range(m):
                if not (target_mask & (1 << i)):
                    next_target = i
                    break
            
            min_cost = float('inf')
            
            # Try assigning each unused num to this target
            for i in range(n):
                if not (used_nums & (1 << i)):
                    cost = costs[i][next_target]
                    remaining_cost = dp(target_mask | (1 << next_target), used_nums | (1 << i))
                    min_cost = min(min_cost, cost + remaining_cost)
            
            return min_cost
        
        return dp(0, 0)