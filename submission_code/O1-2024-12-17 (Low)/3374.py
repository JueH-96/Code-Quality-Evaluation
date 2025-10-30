class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        count = 1  # At least one subarray (the first element itself) if nums is not empty
        curr_len = 1
        
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                curr_len += 1
            else:
                curr_len = 1
            count += curr_len
        
        return count