class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Sort each row in descending order
        for row in grid:
            row.sort(reverse=True)
        
        # Create a set to keep track of used values
        used = set()
        
        # Create a max heap to store the maximum value from each row
        import heapq
        heap = [(-grid[i][0], i, 0) for i in range(m)]
        heapq.heapify(heap)
        
        score = 0
        while heap and len(used) < m:
            val, row, col = heapq.heappop(heap)
            val = -val
            
            if val not in used:
                score += val
                used.add(val)
                
                # Add the next element from the same row if available
                if col + 1 < n:
                    heapq.heappush(heap, (-grid[row][col+1], row, col+1))
        
        return score