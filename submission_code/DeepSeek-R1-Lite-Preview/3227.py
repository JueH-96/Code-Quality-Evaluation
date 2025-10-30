from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        seen = [False] * (n * n + 1)  # Indices 0 to n^2
        duplicate = -1
        # Step 1: Mark seen numbers and find the duplicate
        for row in grid:
            for num in row:
                if seen[num]:
                    duplicate = num
                else:
                    seen[num] = True
        # Step 2: Find the missing number
        for i in range(1, n * n + 1):
            if not seen[i]:
                missing = i
                break
        return [duplicate, missing]