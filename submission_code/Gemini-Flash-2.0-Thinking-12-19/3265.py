class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        found_good_subarray = False
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                if abs(nums[i] - nums[j]) == k:
                    current_sum = sum(nums[i:j+1])
                    if current_sum > max_sum or not found_good_subarray:
                        max_sum = current_sum
                        found_good_subarray = True
        return max_sum