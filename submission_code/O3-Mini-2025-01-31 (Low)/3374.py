class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        res = 0  # will hold the total count of alternating subarrays
        curr_length = 0  # length of the current alternating subarray
        
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                curr_length += 1
            else:
                curr_length = 1  # reset as the current element is same as the previous one
            res += curr_length
        
        return res