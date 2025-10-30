class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def can_zero_out(nums, start, direction):
            n = len(nums)
            curr = start
            nums_copy = nums[:]
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += direction
                else:
                    nums_copy[curr] -= 1
                    direction = -direction
                    curr += direction
            return all(x == 0 for x in nums_copy)
        
        valid_selections = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                if can_zero_out(nums, i, 1) or can_zero_out(nums, i, -1):
                    valid_selections += 1
        
        return valid_selections