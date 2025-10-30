class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        total_sum = 0
        for i in range(len(nums)):
            for j in range(i, min(i + k, len(nums))):
                sub_array = nums[i:j+1]
                total_sum += min(sub_array) + max(sub_array)
        return total_sum