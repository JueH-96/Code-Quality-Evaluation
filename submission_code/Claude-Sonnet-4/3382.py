class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Check all possible subarrays
        for i in range(n):
            max_val = nums[i]
            for j in range(i, n):
                # Update max value as we extend the subarray
                max_val = max(max_val, nums[j])
                
                # Check if first and last elements equal the max
                if nums[i] == nums[j] == max_val:
                    count += 1
        
        return count