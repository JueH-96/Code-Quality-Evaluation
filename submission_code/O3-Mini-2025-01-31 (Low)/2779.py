from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize nums (all uncolored = 0) and the current count of same-color adjacent pairs.
        nums = [0] * n
        curr_adjacent = 0
        result = []
        
        for index, new_color in queries:
            old_color = nums[index]
            
            # For left neighbor: if it exists, adjust count.
            if index - 1 >= 0:
                # If the left neighbor and current index formed a pair before update, remove it.
                if nums[index - 1] == old_color and old_color != 0:
                    curr_adjacent -= 1
                # If after update current and left neighbor form a valid pair, add it.
                if nums[index - 1] == new_color and new_color != 0:
                    curr_adjacent += 1
                    
            # For right neighbor: if it exists, adjust count.
            if index + 1 < n:
                # If the right neighbor and current index formed a pair before update, remove it.
                if nums[index + 1] == old_color and old_color != 0:
                    curr_adjacent -= 1
                # If after update current and right neighbor form a valid pair, add it.
                if nums[index + 1] == new_color and new_color != 0:
                    curr_adjacent += 1
            
            # Update the color at index.
            nums[index] = new_color
            # Append the current count to result after this query.
            result.append(curr_adjacent)
        
        return result