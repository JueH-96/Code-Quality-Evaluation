class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            if len(set(subarray)) >= m:
                max_sum = max(max_sum, sum(subarray))
        return max_sum