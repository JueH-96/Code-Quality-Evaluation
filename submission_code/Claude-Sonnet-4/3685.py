class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        
        # Iterate through all possible starting positions for subarrays of length 3
        for i in range(len(nums) - 2):
            # Check if first + third == half of second
            if nums[i] + nums[i + 2] == nums[i + 1] / 2:
                count += 1
        
        return count