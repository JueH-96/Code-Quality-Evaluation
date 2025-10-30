class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # cost_diff[i] represents the change in cost at position i
        cost_diff = [0] * (k + 2)
        
        for i in range(n // 2):
            a, b = nums[i], nums[n - i - 1]
            diff = abs(a - b)
            max_reach = max(a, b, k - a, k - b)
            
            # Start with cost 2 for all X values
            cost_diff[0] += 2
            
            # For X in [0, max_reach], reduce cost to 1
            cost_diff[0] -= 1
            if max_reach + 1 <= k:
                cost_diff[max_reach + 1] += 1
            
            # For X = diff, reduce cost to 0
            cost_diff[diff] -= 1
            if diff + 1 <= k:
                cost_diff[diff + 1] += 1
        
        # Calculate actual costs using prefix sum
        min_cost = float('inf')
        current_cost = 0
        
        for X in range(k + 1):
            current_cost += cost_diff[X]
            min_cost = min(min_cost, current_cost)
        
        return min_cost