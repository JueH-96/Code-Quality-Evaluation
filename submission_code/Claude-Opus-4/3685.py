class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        
        # Iterate through all possible starting positions for subarrays of length 3
        for i in range(len(nums) - 2):
            # Get the three elements
            first = nums[i]
            second = nums[i + 1]
            third = nums[i + 2]
            
            # Check if first + third equals half of second
            # To avoid floating point, check if 2 * (first + third) == second
            if 2 * (first + third) == second:
                count += 1
        
        return count