from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Function to process and sort a diagonal from (i, j) along the main diagonal.
        def process_diagonal(i, j, reverse: bool):
            diag = []
            r, c = i, j
            while r < n and c < n:
                diag.append(grid[r][c])
                r += 1
                c += 1
            # Sort the diagonal as required
            diag.sort(reverse=reverse)
            # Place sorted values back in the diagonal
            r, c = i, j
            for val in diag:
                grid[r][c] = val
                r += 1
                c += 1

        # Process bottom-left triangle diagonals (including main diagonal)
        # These are diagonals starting from every row at first column.
        # They should be sorted in non-increasing order.
        for i in range(n):
            process_diagonal(i, 0, reverse=True)
            
        # Process top-right triangle diagonals
        # These are diagonals starting from every column in the first row except the first column (already processed).
        # They should be sorted in non-decreasing order.
        for j in range(1, n):
            process_diagonal(0, j, reverse=False)
            
        return grid

# Example usage and tests:
if __name__ == "__main__":
    sol = Solution()
    grid1 = [[1,7,3],[9,8,2],[4,5,6]]
    print(sol.sortMatrix(grid1))  # Expected output: [[8,2,3],[9,6,7],[4,5,1]]
    
    grid2 = [[0,1],[1,2]]
    print(sol.sortMatrix(grid2))  # Expected output: [[2,1],[1,0]]
    
    grid3 = [[1]]
    print(sol.sortMatrix(grid3))  # Expected output: [[1]]