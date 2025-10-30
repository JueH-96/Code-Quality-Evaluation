class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        current_length = 0
        start = -1
        
        for i in range(len(nums)):
            if nums[i] > threshold:
                current_length = 0
                start = -1
            elif nums[i] % 2 == 0:
                if start == -1:
                    start = i
                current_length += 1
            elif start != -1 and nums[i] % 2 != nums[i - 1] % 2:
                current_length += 1
            else:
                current_length = 0
                start = -1
            
            max_length = max(max_length, current_length)
        
        return max_length