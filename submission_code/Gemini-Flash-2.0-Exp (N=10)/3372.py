class Solution:
    def longestMonotonicSubarray(self, nums: list[int]) -> int:
        if not nums:
            return 0
        
        max_len = 1
        
        for i in range(len(nums)):
            
            # Check for increasing subarrays
            curr_len = 1
            for j in range(i + 1, len(nums)):
                if nums[j] > nums[j-1]:
                    curr_len += 1
                else:
                    break
            max_len = max(max_len, curr_len)
            
            # Check for decreasing subarrays
            curr_len = 1
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[j-1]:
                    curr_len += 1
                else:
                    break
            max_len = max(max_len, curr_len)
            
        return max_len