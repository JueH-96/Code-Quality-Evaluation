class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            max_val = nums[i]
            operations = 0
            
            for j in range(i, n):
                if nums[j] < max_val:
                    operations += max_val - nums[j]
                else:
                    max_val = nums[j]
                
                if operations <= k:
                    count += 1
                else:
                    break  # No point checking longer subarrays
        
        return count