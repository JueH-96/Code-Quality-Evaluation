from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        same_adjacent = 0
        answer = []
        
        for index, color in queries:
            old_color = nums[index]
            if old_color == color:
                answer.append(same_adjacent)
                continue
            # Check left neighbor
            if index > 0:
                left_color = nums[index - 1]
                if left_color == old_color and old_color != 0:
                    same_adjacent -= 1
                if left_color == color and color != 0:
                    same_adjacent += 1
            # Check right neighbor
            if index < n - 1:
                right_color = nums[index + 1]
                if right_color == old_color and old_color != 0:
                    same_adjacent -= 1
                if right_color == color and color != 0:
                    same_adjacent += 1
            # Update the color
            nums[index] = color
            # Append current same_adjacent
            answer.append(same_adjacent)
        
        return answer