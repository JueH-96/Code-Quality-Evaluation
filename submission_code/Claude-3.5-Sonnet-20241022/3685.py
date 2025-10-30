class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate through all possible subarrays of length 3
        for i in range(n-2):
            # Get the three elements
            first = nums[i]
            middle = nums[i+1]
            third = nums[i+2]
            
            # Check if sum of first and third equals half of middle
            if first + third == middle/2:
                count += 1
                
        return count