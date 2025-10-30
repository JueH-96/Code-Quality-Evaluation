class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        length = 1  # At least each element is a subarray itself
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                length += 1
            else:
                count += (length * (length + 1)) // 2
                length = 1
        
        # Add the subarrays from the last sequence
        count += (length * (length + 1)) // 2
        
        return count