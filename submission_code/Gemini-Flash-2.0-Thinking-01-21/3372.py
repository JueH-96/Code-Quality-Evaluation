class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_length = 0
        for i in range(n):
            current_length_inc = 1
            j = i + 1
            while j < n and nums[j] > nums[j - 1]:
                current_length_inc += 1
                j += 1

            current_length_dec = 1
            j = i + 1
            while j < n and nums[j] < nums[j - 1]:
                current_length_dec += 1
                j += 1
            
            max_length = max(max_length, current_length_inc, current_length_dec)
        return max_length