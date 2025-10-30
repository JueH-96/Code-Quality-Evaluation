from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        current_count = 0
        answer = []
        
        for idx, color in queries:
            prev_color = nums[idx]
            if prev_color == color:
                answer.append(current_count)
                continue
            
            # Subtract old pairs formed by prev_color
            if prev_color != 0:
                # Left neighbor
                if idx > 0 and nums[idx - 1] == prev_color:
                    current_count -= 1
                # Right neighbor
                if idx < n - 1 and nums[idx + 1] == prev_color:
                    current_count -= 1
            
            # Add new pairs formed by the new color
            # Left neighbor
            if idx > 0 and nums[idx - 1] == color:
                current_count += 1
            # Right neighbor
            if idx < n - 1 and nums[idx + 1] == color:
                current_count += 1
            
            # Update the color of the current index
            nums[idx] = color
            answer.append(current_count)
        
        return answer