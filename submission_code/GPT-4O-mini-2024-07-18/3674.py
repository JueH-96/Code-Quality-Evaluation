class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        # Two pointers for the sliding window
        left = 0
        total_increments = 0
        
        for right in range(n):
            # If the current element is less than the previous one,
            # we need to calculate how many increments are needed
            if right > 0 and nums[right] < nums[right - 1]:
                total_increments += nums[right - 1] - nums[right]
            
            # While the total increments exceed k, move the left pointer
            while total_increments > k:
                if left < right and nums[left] < nums[left + 1]:
                    total_increments -= nums[left + 1] - nums[left]
                left += 1
            
            # Count the number of valid subarrays ending at 'right'
            count += (right - left + 1)
        
        return count