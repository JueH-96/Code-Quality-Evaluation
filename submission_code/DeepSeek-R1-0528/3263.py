class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        for i in range(1, n-1):
            for j in range(i+1, n):
                total = nums[0] + nums[i] + nums[j]
                if total < min_cost:
                    min_cost = total
        return min_cost