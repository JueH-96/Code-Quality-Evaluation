from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        result = []
        for query in queries:
            index, color = query
            old_color = nums[index]
            if old_color == color:
                result.append(count)
                continue
            # Remove contributions from old color
            if index > 0 and nums[index - 1] == old_color:
                count -= 1
            if index < n - 1 and nums[index + 1] == old_color:
                count -= 1
            # Update to new color
            nums[index] = color
            # Add contributions from new color
            if index > 0 and nums[index - 1] == color:
                count += 1
            if index < n - 1 and nums[index + 1] == color:
                count += 1
            result.append(count)
        return result