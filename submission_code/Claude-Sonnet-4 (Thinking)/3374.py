class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        result = 0
        current_length = 1
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                current_length += 1
            else:
                result += current_length * (current_length + 1) // 2
                current_length = 1
        
        # Add contribution of the last segment
        result += current_length * (current_length + 1) // 2
        
        return result