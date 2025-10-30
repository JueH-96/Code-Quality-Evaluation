class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        
        # Check every subarray of length 3
        for i in range(len(nums) - 2):
            first = nums[i]
            second = nums[i + 1]
            third = nums[i + 2]
            
            # Check if first + third = second/2
            # Rearranged as: 2 * (first + third) = second
            if 2 * (first + third) == second:
                count += 1
        
        return count