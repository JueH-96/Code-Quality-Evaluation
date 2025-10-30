from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        total = 0
        answer = []
        
        for index, color in queries:
            old_color = nums[index]
            if old_color == color:
                answer.append(total)
                continue
            
            # Subtract old contributions
            if index > 0:
                left = nums[index - 1]
                if old_color != 0 and left == old_color:
                    total -= 1
            if index < n - 1:
                right = nums[index + 1]
                if old_color != 0 and right == old_color:
                    total -= 1
            
            # Update the color
            nums[index] = color
            
            # Add new contributions
            if index > 0:
                left = nums[index - 1]
                if left == color and left != 0:
                    total += 1
            if index < n - 1:
                right = nums[index + 1]
                if right == color and right != 0:
                    total += 1
            
            answer.append(total)
        
        return answer