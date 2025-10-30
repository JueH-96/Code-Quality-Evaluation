from typing import List

class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n = len(grid)
        # Create a list of all elements with their row index
        elements = []
        for i in range(n):
            for j in range(len(grid[i])):
                elements.append((grid[i][j], i))
        
        # Sort elements in descending order of value
        elements.sort(key=lambda x: x[0], reverse=True)
        
        # Initialize count per row to keep track of elements taken from each row
        count_row = [0] * n
        current_sum = 0
        current_count = 0
        
        # Iterate through the sorted elements
        for val, row in elements:
            if count_row[row] < limits[row] and current_count < k:
                current_sum += val
                count_row[row] += 1
                current_count += 1
                if current_count == k:
                    break  # No need to check further elements
        
        return current_sum