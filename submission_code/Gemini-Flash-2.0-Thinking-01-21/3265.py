class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        found_good_subarray = False
        n = len(nums)
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += nums[j]
                if abs(nums[i] - nums[j]) == k:
                    found_good_subarray = True
                    max_sum = max(max_sum, current_sum)
        if not found_good_subarray:
            return 0
        return max_sum