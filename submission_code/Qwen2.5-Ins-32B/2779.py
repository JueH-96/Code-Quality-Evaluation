from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            # Check if the current index and its neighbors were previously the same color
            if index > 0 and nums[index] == nums[index - 1] and nums[index] != 0:
                count -= 1
            if index < n - 1 and nums[index] == nums[index + 1] and nums[index] != 0:
                count -= 1
            
            # Color the current index
            nums[index] = color
            
            # Check if the current index and its neighbors are now the same color
            if index > 0 and nums[index] == nums[index - 1]:
                count += 1
            if index < n - 1 and nums[index] == nums[index + 1]:
                count += 1
            
            answer.append(count)
        
        return answer