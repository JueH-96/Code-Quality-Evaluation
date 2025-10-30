from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize array with 0 (uncolored)
        nums = [0] * n
        # This variable will track the number of adjacent same-colored pairs.
        same_adjacent = 0
        results = []
        
        # Process each query
        for idx, new_color in queries:
            old_color = nums[idx]
            
            # If the index already has the same color, nothing changes.
            if old_color == new_color:
                results.append(same_adjacent)
                continue
            
            # Check the left neighbor if it exists.
            if idx - 1 >= 0:
                # If the left neighbor formed a pair with the old color, remove that pair.
                if nums[idx - 1] == old_color and old_color != 0:
                    same_adjacent -= 1
                # If the left neighbor will form a pair with the new color, add that pair.
                if nums[idx - 1] == new_color and new_color != 0:
                    same_adjacent += 1
            
            # Check the right neighbor if it exists.
            if idx + 1 < n:
                # If the right neighbor formed a pair with the old color, remove that pair.
                if nums[idx + 1] == old_color and old_color != 0:
                    same_adjacent -= 1
                # If the right neighbor will form a pair with the new color, add that pair.
                if nums[idx + 1] == new_color and new_color != 0:
                    same_adjacent += 1
            
            # Update the color at the current index.
            nums[idx] = new_color
            
            # Append the current count of same adjacent pairs to the results.
            results.append(same_adjacent)
        
        return results