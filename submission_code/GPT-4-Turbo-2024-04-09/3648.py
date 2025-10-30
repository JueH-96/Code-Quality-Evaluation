class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        from functools import lru_cache
        
        n = len(fruits)
        
        @lru_cache(None)
        def dp(x1, y1, x2, y2, x3, y3):
            # Calculate the current position of the third child based on the other two
            x3 = x1 + y1 - x2 - y2 + x3
            y3 = x1 + y1 - x2 - y2 + y3
            
            # Check if any child is out of bounds
            if not (0 <= x1 < n and 0 <= y1 < n and 0 <= x2 < n and 0 <= y2 < n and 0 <= x3 < n and 0 <= y3 < n):
                return float('-inf')
            
            # If all children reached the bottom-right corner
            if x1 == x2 == x3 == y1 == y2 == y3 == n-1:
                return fruits[n-1][n-1]
            
            # Calculate the fruits collected at the current positions
            current_fruits = 0
            positions = {(x1, y1), (x2, y2), (x3, y3)}
            for x, y in positions:
                current_fruits += fruits[x][y]
            
            # Calculate the maximum fruits collected by trying all possible moves
            result = 0
            moves = [(1, 0), (1, 1), (0, 1)]
            for dx1, dy1 in moves:
                for dx2, dy2 in moves:
                    result = max(result, dp(x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2, x3, y3))
            
            return current_fruits + result
        
        # Start the recursion from the initial positions of the children
        return dp(0, 0, 0, n-1, n-1, 0)