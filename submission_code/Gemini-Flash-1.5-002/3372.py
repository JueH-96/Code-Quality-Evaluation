class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_len = 1
        curr_len = 1
        
        # Check for increasing subarrays
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                curr_len += 1
            else:
                max_len = max(max_len, curr_len)
                curr_len = 1
        max_len = max(max_len, curr_len)
        
        curr_len = 1
        #Check for decreasing subarrays
        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                curr_len += 1
            else:
                max_len = max(max_len, curr_len)
                curr_len = 1
        max_len = max(max_len, curr_len)

        return max_len