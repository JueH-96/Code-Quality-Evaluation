from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        colors = [0] * n
        current = 0
        answer = []
        
        for index, color in queries:
            old_color = colors[index]
            
            # Subtract old contributions if old_color was non-zero
            if old_color != 0:
                # Check left neighbor
                if index > 0 and colors[index - 1] == old_color:
                    current -= 1
                # Check right neighbor
                if index < n - 1 and colors[index + 1] == old_color:
                    current -= 1
            
            # Update the color
            colors[index] = color
            
            # Add new contributions
            # Check left neighbor
            if index > 0 and colors[index - 1] == color:
                current += 1
            # Check right neighbor
            if index < n - 1 and colors[index + 1] == color:
                current += 1
            
            answer.append(current)
        
        return answer