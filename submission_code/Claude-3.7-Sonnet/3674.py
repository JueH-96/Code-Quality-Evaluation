class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            operations = 0
            current_val = nums[i]  # This tracks what the next element should be at least
            count += 1  # Single element subarrays are always non-decreasing
            
            for j in range(i+1, n):
                if nums[j] < current_val:
                    # If current element is smaller than previous, we need operations
                    operations += current_val - nums[j]
                    if operations > k:
                        break  # Can't make this subarray non-decreasing with k operations
                else:
                    # If current element is already >= previous, update current_val
                    current_val = nums[j]
                
                count += 1  # This subarray can be made non-decreasing
            
        return count