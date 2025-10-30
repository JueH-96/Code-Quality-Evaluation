class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        n, m = len(grid), len(grid[0])
        
        # Flatten the grid and store (value, row, col)
        elements = []
        for i in range(n):
            for j in range(m):
                elements.append((grid[i][j], i, j))
        
        # Sort elements by value in descending order
        elements.sort(reverse=True)
        
        # Keep track of how many elements we've taken from each row
        row_counts = [0] * n
        total_sum = 0
        elements_taken = 0
        
        # Greedily take the highest value elements
        for value, row, col in elements:
            # Check if we can take this element
            if row_counts[row] < limits[row] and elements_taken < k:
                total_sum += value
                row_counts[row] += 1
                elements_taken += 1
        
        return total_sum