class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Dictionary to store diagonals
        diagonals = {}
        
        # Collect all elements by their diagonal
        for i in range(n):
            for j in range(n):
                diagonal_sum = i + j
                if diagonal_sum not in diagonals:
                    diagonals[diagonal_sum] = []
                diagonals[diagonal_sum].append((i, j, grid[i][j]))
        
        # Sort each diagonal
        for diagonal_sum in diagonals:
            if diagonal_sum >= n - 1:
                # Bottom-left triangle (including main diagonal) - sort in non-increasing order
                diagonals[diagonal_sum].sort(key=lambda x: x[2], reverse=True)
            else:
                # Top-right triangle - sort in non-decreasing order
                diagonals[diagonal_sum].sort(key=lambda x: x[2])
        
        # Create result matrix
        result = [[0] * n for _ in range(n)]
        
        # Place sorted elements back
        for diagonal_sum in diagonals:
            # Sort positions by row index to maintain diagonal order
            positions = sorted([(i, j) for i, j, _ in diagonals[diagonal_sum]])
            values = [val for _, _, val in diagonals[diagonal_sum]]
            
            for idx, (i, j) in enumerate(positions):
                result[i][j] = values[idx]
        
        return result