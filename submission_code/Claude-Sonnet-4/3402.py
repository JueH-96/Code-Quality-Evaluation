class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        if n == 1:
            return 0
            
        max_val = max(nums)
        total_increases = sum(max_val - num for num in nums)
        max_increases = max_val - min(nums)
        
        # If cost2 >= 2*cost1, always use operation 1
        if cost2 >= 2 * cost1:
            return (total_increases * cost1) % MOD
        
        min_cost = float('inf')
        
        # Try different target values starting from max_val
        # We might need to increase the target if the max element needs too many increases
        for target in range(max_val, max_val + max_increases + 1):
            total_inc = sum(target - num for num in nums)
            max_inc = target - min(nums)
            
            # Case 1: Use only operation 1
            cost_op1_only = total_inc * cost1
            min_cost = min(min_cost, cost_op1_only)
            
            # Case 2: Use operation 2 as much as possible
            if total_inc >= 2 * max_inc:
                # We can pair all increases of max element with others
                pairs = total_inc // 2
                remaining = total_inc % 2
                cost_mixed = pairs * cost2 + remaining * cost1
            else:
                # Max element needs more increases than we can pair
                pairs = total_inc - max_inc
                remaining = max_inc - pairs
                cost_mixed = pairs * cost2 + remaining * cost1
            
            min_cost = min(min_cost, cost_mixed)
        
        return min_cost % MOD