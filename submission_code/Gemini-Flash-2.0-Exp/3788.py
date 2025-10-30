class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                subarray = nums[i:j+1]
                if len(set(subarray)) == len(subarray):
                    max_sum = max(max_sum, sum(subarray))
        return max_sum