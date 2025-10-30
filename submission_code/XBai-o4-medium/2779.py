from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        current_count = 0
        answer = []
        
        for idx, color in queries:
            old_color = nums[idx]
            # Subtract the old contributions if the old color was non-zero
            if old_color != 0:
                # Check left neighbor
                if idx > 0 and nums[idx - 1] == old_color:
                    current_count -= 1
                # Check right neighbor
                if idx < n - 1 and nums[idx + 1] == old_color:
                    current_count -= 1
            # Update the color
            nums[idx] = color
            new_color = color
            # Add new contributions
            # Check left neighbor
            if idx > 0 and nums[idx - 1] == new_color:
                current_count += 1
            # Check right neighbor
            if idx < n - 1 and nums[idx + 1] == new_color:
                current_count += 1
            answer.append(current_count)
        
        return answer