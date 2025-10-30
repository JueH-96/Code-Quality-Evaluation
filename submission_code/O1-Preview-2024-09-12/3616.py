class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        
        zero_indices = [i for i, val in enumerate(nums) if val == 0]
        
        for start in zero_indices:
            for dir in [-1, 1]:  # -1 for left, +1 for right
                nums_copy = nums[:]  # Make a copy of nums
                curr = start
                curr_dir = dir
                while 0 <= curr < n:
                    if nums_copy[curr] == 0:
                        # Move one step in the current direction
                        curr += curr_dir
                    elif nums_copy[curr] > 0:
                        # Decrement nums[curr], reverse direction, and move one step
                        nums_copy[curr] -= 1
                        curr_dir = -curr_dir  # Reverse direction
                        curr += curr_dir  # Move one step in new direction
                # After process ends, check if nums_copy is all zeros
                if all(val == 0 for val in nums_copy):
                    result += 1
        return result