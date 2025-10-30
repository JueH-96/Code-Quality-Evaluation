class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        current_length = -1
        
        for i in range(1, len(nums)):
            if current_length > 0 and nums[i] == nums[i - 2]:
                current_length += 1
            else:
                if nums[i] == nums[i - 1] + 1:
                    current_length = 2
                else:
                    current_length = -1
            max_length = max(max_length, current_length)
        
        return max_length