class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        import heapq
        
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        # Sort each row in descending order
        sorted_rows = []
        for i in range(n):
            sorted_row = sorted(grid[i], reverse=True)
            sorted_rows.append(sorted_row)
        
        # Create a max heap with (negative value, row_index, col_index)
        # We use negative values because Python has min heap by default
        heap = []
        
        # Initialize heap with the largest element from each row (if limit allows)
        for i in range(n):
            if limits[i] > 0 and m > 0:
                heapq.heappush(heap, (-sorted_rows[i][0], i, 0))
        
        total_sum = 0
        elements_taken = 0
        
        # Keep track of how many elements taken from each row
        taken_from_row = [0] * n
        
        while heap and elements_taken < k:
            # Get the largest available element
            neg_val, row, col = heapq.heappop(heap)
            val = -neg_val
            
            # Add to sum
            total_sum += val
            elements_taken += 1
            taken_from_row[row] += 1
            
            # If we can take more from this row and there are more elements
            next_col = col + 1
            if taken_from_row[row] < limits[row] and next_col < m:
                heapq.heappush(heap, (-sorted_rows[row][next_col], row, next_col))
        
        return total_sum