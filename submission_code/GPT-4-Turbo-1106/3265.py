class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        current_sum = 0
        left = 0
        
        for right in range(len(nums)):
            current_sum += nums[right]
            
            # Check if the current subarray is good
            if abs(nums[right] - nums[left]) == k:
                max_sum = max(max_sum, current_sum)
                
                # Move the left pointer to find the next good subarray
                current_sum -= nums[left]
                left += 1
            
            # If the difference is greater than k, move the left pointer
            while left < right and abs(nums[right] - nums[left]) > k:
                current_sum -= nums[left]
                left += 1
        
        return max_sum