class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        current_length = 1
        
        for i in range(1, len(nums)):
            if (i % 2 == 1 and nums[i] == nums[i - 1] + 1) or (i % 2 == 0 and nums[i] == nums[i - 1] - 1):
                current_length += 1
            else:
                if current_length > 1:
                    max_length = max(max_length, current_length)
                current_length = 1 if i % 2 == 0 or nums[i] != nums[i - 1] + 1 else 2
        
        if current_length > 1:
            max_length = max(max_length, current_length)
        
        return max_length