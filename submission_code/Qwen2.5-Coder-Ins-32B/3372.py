class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_length = 1
        current_length = 1
        increasing = None
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                if increasing is False:
                    current_length = 1
                increasing = True
                current_length += 1
            elif nums[i] < nums[i - 1]:
                if increasing is True:
                    current_length = 1
                increasing = False
                current_length += 1
            else:
                current_length = 1
                increasing = None
            
            max_length = max(max_length, current_length)
        
        return max_length