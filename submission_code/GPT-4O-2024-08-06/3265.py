class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        n = len(nums)
        
        # Iterate over all possible starting points of subarrays
        for i in range(n):
            # Iterate over all possible ending points of subarrays starting from i
            for j in range(i + 1, n):
                # Check if the subarray nums[i..j] is good
                if abs(nums[i] - nums[j]) == k:
                    # Calculate the sum of the subarray nums[i..j]
                    current_sum = sum(nums[i:j+1])
                    # Update max_sum if the current subarray sum is greater
                    max_sum = max(max_sum, current_sum)
        
        return max_sum