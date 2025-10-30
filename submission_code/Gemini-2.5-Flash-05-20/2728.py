from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        total_score = 0
        
        # Step 1: Sort each row in ascending order.
        # This makes it easy to retrieve the largest element (always at the end)
        # and remove it efficiently using list.pop().
        for row in nums:
            row.sort()
            
        # Determine the number of columns.
        # The problem implies a rectangular matrix, so all rows have the same length.
        # The number of operations (rounds) will be equal to the number of columns,
        # as one element is removed from each row per round until all elements are gone.
        num_columns = len(nums[0])
        num_rows = len(nums)
        
        # Step 2: Perform operations round by round.
        # Each iteration of this loop represents one "round" of removals.
        for _ in range(num_columns):
            current_round_max = 0
            
            # From each row, select the largest number and remove it.
            for r_idx in range(num_rows):
                # Since the row is sorted in ascending order, nums[r_idx].pop()
                # removes and returns the largest element from the end of the list.
                removed_num = nums[r_idx].pop()
                
                # Identify the highest number amongst all those removed in this step.
                current_round_max = max(current_round_max, removed_num)
            
            # Add that highest number from the current round to the total score.
            total_score += current_round_max
            
        # Return the final accumulated score.
        return total_score