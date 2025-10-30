class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = float('-inf')
        found_good = False
        
        for i in range(n):
            current_sum = nums[i]
            for j in range(i + 1, n):
                current_sum += nums[j]
                # Check if this is a good subarray
                if abs(nums[i] - nums[j]) == k:
                    max_sum = max(max_sum, current_sum)
                    found_good = True
        
        return max_sum if found_good else 0