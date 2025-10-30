class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Calculate the AND of the entire array - this is the minimum possible sum
        total_and = nums[0]
        for num in nums[1:]:
            total_and &= num
        
        # If the minimum sum is not 0, we can only have 1 subarray (the entire array)
        if total_and != 0:
            return 1
        
        # If minimum sum is 0, greedily create subarrays with score 0
        count = 0
        i = 0
        
        while i < len(nums):
            current_and = nums[i]
            j = i
            
            # Extend current subarray until AND becomes 0
            while current_and != 0 and j < len(nums):
                current_and &= nums[j]
                j += 1
            
            count += 1
            i = j
        
        return count