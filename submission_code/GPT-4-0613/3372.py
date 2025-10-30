class Solution:
    def longestMonotonicSubarray(self, nums):
        if len(nums) < 2:
            return len(nums)
        
        max_len = 1
        cur_len = 1
        increasing = None
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                if increasing is None or increasing:
                    cur_len += 1
                    increasing = True
                else:
                    cur_len = 2
                    increasing = True
            elif nums[i] < nums[i-1]:
                if increasing is None or not increasing:
                    cur_len += 1
                    increasing = False
                else:
                    cur_len = 2
                    increasing = False
            else:
                cur_len = 1
                increasing = None
                
            max_len = max(max_len, cur_len)
            
        return max_len