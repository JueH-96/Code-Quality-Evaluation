class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        valid_count = 0
        
        # Try each position where nums[i] == 0 as starting position
        for start in range(n):
            if nums[start] == 0:
                # Try both initial directions: -1 for left, 1 for right
                for initial_dir in [-1, 1]:
                    # Make a copy of nums for simulation
                    nums_copy = nums[:]
                    curr = start
                    direction = initial_dir
                    
                    # Simulate the process
                    while 0 <= curr < n:
                        if nums_copy[curr] == 0:
                            # Move in the current direction
                            curr += direction
                        else:  # nums_copy[curr] > 0
                            # Decrement the value
                            nums_copy[curr] -= 1
                            # Reverse direction
                            direction = -direction
                            # Take a step in the new direction
                            curr += direction
                    
                    # Check if all elements are 0
                    if all(x == 0 for x in nums_copy):
                        valid_count += 1
        
        return valid_count