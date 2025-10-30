class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                if len(sub_array) % k == 0:
                    max_sum = max(max_sum, sum(sub_array))
        return max_sum