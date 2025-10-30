from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        adjacency_count = 0
        answer = []
        
        for idx, color in queries:
            old_color = nums[idx]
            
            # If the color doesn't change, no need to re-check adjacencies
            if old_color == color:
                answer.append(adjacency_count)
                continue
            
            # Remove any adjacency formed by the old color
            if idx > 0 and nums[idx - 1] == old_color and old_color != 0:
                adjacency_count -= 1
            if idx < n - 1 and nums[idx + 1] == old_color and old_color != 0:
                adjacency_count -= 1
            
            # Update to the new color
            nums[idx] = color
            
            # Add new adjacency if formed
            if idx > 0 and nums[idx - 1] == color and color != 0:
                adjacency_count += 1
            if idx < n - 1 and nums[idx + 1] == color and color != 0:
                adjacency_count += 1
            
            answer.append(adjacency_count)
        
        return answer