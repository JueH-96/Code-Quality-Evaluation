class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        length = 1  # length of the current alternating subarray ending at the current position
        
        for i in range(len(nums)):
            if i > 0 and nums[i] != nums[i-1]:
                length += 1
            else:
                length = 1
            
            count += length
        
        return count