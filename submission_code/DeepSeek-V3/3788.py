class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        memo = {}
        
        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if j < i:
                return 0
            if j - i + 1 == 1:
                memo[(i, j)] = nums[i]
                return nums[i]
            if j - i + 1 == 2:
                memo[(i, j)] = max(nums[i], nums[j])
                return max(nums[i], nums[j])
            # Case 1: remove i and i+1
            cost1 = max(nums[i], nums[i+1]) + dp(i+2, j)
            # Case 2: remove i and i+2
            cost2 = max(nums[i], nums[i+2])
            remaining_cost = 0
            if i+1 <= j:
                remaining_cost += nums[i+1]
            if i+3 <= j:
                remaining_cost += dp(i+3, j)
            cost2 += remaining_cost
            # Case 3: remove i+1 and i+2
            cost3 = max(nums[i+1], nums[i+2])
            remaining_cost3 = 0
            if i <= j:
                remaining_cost3 += nums[i]
            if i+3 <= j:
                remaining_cost3 += dp(i+3, j)
            cost3 += remaining_cost3
            memo[(i, j)] = min(cost1, cost2, cost3)
            return memo[(i, j)]
        
        return dp(0, n-1)