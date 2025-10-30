class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try each position where nums[i] == 0 as starting position
        for start_pos in range(n):
            if nums[start_pos] != 0:
                continue
            
            # Try both directions
            for direction in [-1, 1]:  # -1 for left, 1 for right
                # Make a copy of nums for simulation
                test_nums = nums[:]
                curr = start_pos
                curr_direction = direction
                
                while 0 <= curr < n:
                    if test_nums[curr] == 0:
                        # Move in current direction
                        curr += curr_direction
                    else:
                        # Decrement and reverse direction
                        test_nums[curr] -= 1
                        curr_direction = -curr_direction
                        curr += curr_direction
                
                # Check if all elements are 0
                if all(x == 0 for x in test_nums):
                    count += 1
        
        return count