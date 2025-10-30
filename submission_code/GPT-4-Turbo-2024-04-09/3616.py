class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        from collections import deque
        
        def simulate(nums, start, direction):
            n = len(nums)
            current = start
            counts = nums[:]
            move_right = direction == 'right'
            
            while 0 <= current < n:
                if counts[current] == 0:
                    if move_right:
                        current += 1
                    else:
                        current -= 1
                else:
                    counts[current] -= 1
                    move_right = not move_right
                    if move_right:
                        current += 1
                    else:
                        current -= 1
            
            return all(x == 0 for x in counts)
        
        valid_selections = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                # Try moving left
                if simulate(nums, i, 'left'):
                    valid_selections += 1
                # Try moving right
                if simulate(nums, i, 'right'):
                    valid_selections += 1
        
        return valid_selections