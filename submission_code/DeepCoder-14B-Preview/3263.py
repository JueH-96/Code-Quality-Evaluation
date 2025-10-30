class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                current = nums[0] + nums[i + 1] + nums[j + 1]
                if current < min_sum:
                    min_sum = current
        return min_sum