class Solution:
    def minimumSum(self, nums):
        n = len(nums)
        min_sum = float('inf')
        for j in range(1, n-1):
            left = [i for i in nums[:j] if i < nums[j]]
            right = [k for k in nums[j+1:] if k < nums[j]]
            if left and right:
                min_sum = min(min_sum, nums[j] + min(left) + min(right))
        return min_sum if min_sum != float('inf') else -1