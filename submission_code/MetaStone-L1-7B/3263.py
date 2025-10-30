class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                current_sum = nums[0] + nums[i + 1] + nums[j + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
        return min_sum