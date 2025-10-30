class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        min_sum = float('inf')
        n = len(nums)
        for j in range(1, n - 1):
            for i in range(j):
                for k in range(j + 1, n):
                    if nums[i] < nums[j] and nums[k] < nums[j]:
                        min_sum = min(min_sum, nums[i] + nums[j] + nums[k])
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum