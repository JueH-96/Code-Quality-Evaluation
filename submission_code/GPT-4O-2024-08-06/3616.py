from typing import List

class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def can_zero_all(start, direction):
            n = len(nums)
            nums_copy = nums[:]
            curr = start
            dir_step = 1 if direction == 'right' else -1
            
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += dir_step
                else:
                    nums_copy[curr] -= 1
                    dir_step = -dir_step
                    curr += dir_step
            
            return all(x == 0 for x in nums_copy)
        
        valid_selections = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                if can_zero_all(i, 'left'):
                    valid_selections += 1
                if can_zero_all(i, 'right'):
                    valid_selections += 1
        
        return valid_selections