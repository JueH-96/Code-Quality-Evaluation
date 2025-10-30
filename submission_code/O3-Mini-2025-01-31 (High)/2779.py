from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        ans = []
        colors = [0] * n  # initially uncolored
        same_count = 0   # count for adjacent same-colored pairs
        
        for pos, new_color in queries:
            old_color = colors[pos]
            # If the new color is the same as the old color, no changes occur.
            if old_color == new_color:
                ans.append(same_count)
                continue
            
            # Check the left neighbor.
            if pos - 1 >= 0:
                left_color = colors[pos - 1]
                # If the left neighbor was connected with the old color, remove that connection.
                if left_color == old_color and old_color != 0:
                    same_count -= 1
                # If the left neighbor matches the new color, add a connection.
                if left_color == new_color:
                    same_count += 1
                    
            # Check the right neighbor.
            if pos + 1 < n:
                right_color = colors[pos + 1]
                # If the right neighbor was connected with the old color, remove that connection.
                if right_color == old_color and old_color != 0:
                    same_count -= 1
                # If the right neighbor matches the new color, add a connection.
                if right_color == new_color:
                    same_count += 1
            
            # Update the color for the current index.
            colors[pos] = new_color
            # Append the current count of adjacent same-colored pairs.
            ans.append(same_count)
        
        return ans