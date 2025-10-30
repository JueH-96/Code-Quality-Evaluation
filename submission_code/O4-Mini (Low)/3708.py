from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        """
        Traverse the grid in a zigzag pattern (right on even-indexed rows,
        left on odd-indexed rows), and skip every alternate cell in the
        traversal sequence.
        """
        result = []
        take = True  # Flag to decide whether to include the current cell
        
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        for r in range(rows):
            # Determine traversal direction for this row
            if r % 2 == 0:
                # left to right
                col_range = range(cols)
            else:
                # right to left
                col_range = range(cols - 1, -1, -1)
            
            # Traverse the row in the chosen direction
            for c in col_range:
                if take:
                    result.append(grid[r][c])
                # Flip the flag after each visited cell
                take = not take
        
        return result

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.zigzagTraversal([[1, 2], [3, 4]]))          # [1, 4]
    print(sol.zigzagTraversal([[2, 1], [2, 1], [2, 1]]))  # [2, 1, 2]
    print(sol.zigzagTraversal([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # [1, 3, 5, 7, 9]