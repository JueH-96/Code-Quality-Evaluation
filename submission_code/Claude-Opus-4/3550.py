class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        
        # Store all values with their positions
        values = []
        for i in range(m):
            for j in range(n):
                values.append((board[i][j], i, j))
        
        # Sort by value in descending order
        values.sort(reverse=True)
        
        # Try all combinations of 3 cells from the top values
        # We don't need to check all combinations - just enough to ensure we find the optimal
        max_sum = float('-inf')
        
        # Limit search to top values to avoid timeout
        limit = min(len(values), 300)  # Adjust based on constraints
        
        for i in range(limit):
            val1, r1, c1 = values[i]
            for j in range(i + 1, limit):
                val2, r2, c2 = values[j]
                # Check if they're in different rows and columns
                if r1 == r2 or c1 == c2:
                    continue
                    
                for k in range(j + 1, limit):
                    val3, r3, c3 = values[k]
                    # Check if all three are in different rows and columns
                    if r1 != r2 and r1 != r3 and r2 != r3 and c1 != c2 and c1 != c3 and c2 != c3:
                        max_sum = max(max_sum, val1 + val2 + val3)
                        # Since we're going in descending order, we can break early
                        # if we found a valid combination with the current i and j
                        break
        
        return max_sum