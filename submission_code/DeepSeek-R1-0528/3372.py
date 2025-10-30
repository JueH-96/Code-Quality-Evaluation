class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_inc = 1
        max_dec = 1
        cur_inc = 1
        cur_dec = 1
        
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                cur_inc += 1
                cur_dec = 1
            elif nums[i] < nums[i-1]:
                cur_dec += 1
                cur_inc = 1
            else:
                cur_inc = 1
                cur_dec = 1
            
            if cur_inc > max_inc:
                max_inc = cur_inc
            if cur_dec > max_dec:
                max_dec = cur_dec
        
        return max(max_inc, max_dec)