class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()  # Sort to process minimum elements first
        
        # Process in pairs - for each pair (a,b), append them as (b,a)
        for i in range(0, len(nums), 2):
            nums[i], nums[i+1] = nums[i+1], nums[i]  # Swap adjacent elements
        
        return nums