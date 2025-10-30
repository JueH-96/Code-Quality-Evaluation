class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        # For each starting position
        for i in range(n):
            operations = 0
            max_val = 0
            
            # Extend subarray to the right
            for j in range(i, n):
                # For position j, we need it to be at least max_val
                max_val = max(max_val, nums[j])
                operations += max_val - nums[j]
                
                # If operations exceed k, break
                if operations > k:
                    break
                    
                # This subarray [i:j+1] can be made non-decreasing
                count += 1
        
        return count