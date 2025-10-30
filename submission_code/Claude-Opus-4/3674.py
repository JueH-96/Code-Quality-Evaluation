class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            max_val = nums[i]
            operations = 0
            
            for j in range(i, n):
                # Update maximum value in current subarray
                max_val = max(max_val, nums[j])
                
                # Add operations needed to bring nums[j] to max_val
                operations += max_val - nums[j]
                
                # If we exceed k operations, we can't make any longer subarrays
                # starting from i non-decreasing
                if operations > k:
                    break
                
                # This subarray [i, j] can be made non-decreasing
                count += 1
        
        return count