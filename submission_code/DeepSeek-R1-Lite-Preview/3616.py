class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        from typing import List

        n = len(nums)
        original_nums = nums.copy()
        count = 0
        # Find all starting positions where nums[curr] == 0
        starting_positions = [i for i in range(n) if original_nums[i] == 0]
        # Directions: 'left' and 'right'
        directions = ['left', 'right']
        # Simulate for each starting position and direction
        for pos in starting_positions:
            for direction in directions:
                # Make a copy of original nums for this simulation
                temp_nums = original_nums.copy()
                curr = pos
                move_direction = direction
                # Simulate the process
                while 0 <= curr < n:
                    if temp_nums[curr] == 0:
                        # Move in current direction
                        if move_direction == 'right':
                            curr += 1
                        else:
                            curr -= 1
                    else:
                        # Decrement temp_nums[curr] by 1
                        temp_nums[curr] -= 1
                        # Reverse direction
                        if move_direction == 'right':
                            move_direction = 'left'
                            # Move left
                            curr -= 1
                        else:
                            move_direction = 'right'
                            # Move right
                            curr += 1
                # After simulation, check if all temp_nums are zero
                if all(x == 0 for x in temp_nums):
                    count += 1
        return count