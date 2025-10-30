class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if nums[i] < nums[j] and nums[k] < nums[j]:
                        min_sum = min(min_sum, nums[i] + nums[j] + nums[k])
        if min_sum == float('inf'):
            return -1
        return min_sum