class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We need to split nums into three contiguous parts:
        # [0..i-1], [i..j-1], and [j..n-1].
        # The cost is nums[0] + nums[i] + nums[j].
        # We'll just iterate over all valid i, j to find the minimum possible cost.
        
        min_cost = float('inf')
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cost = nums[0] + nums[i] + nums[j]
                if cost < min_cost:
                    min_cost = cost
                    
        return min_cost