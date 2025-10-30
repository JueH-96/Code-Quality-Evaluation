from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # There are 4 possible 2x2 sub-squares in a 3x3 grid.
        # Their top-left coordinates are: (0,0), (0,1), (1,0), and (1,1)
        for i in range(2):
            for j in range(2):
                # Gather the colors in the current 2x2 sub-square.
                square = [grid[i][j], grid[i][j+1],
                          grid[i+1][j], grid[i+1][j+1]]
                
                # Count the number of black and white cells.
                countB = square.count('B')
                countW = square.count('W')
                
                # If three or all four cells are the same color,
                # then flipping the one different cell (if any) will make the square uniform.
                if countB >= 3 or countW >= 3:
                    return True
        return False

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    grid1 = [["B","W","B"], ["B","W","W"], ["B","W","B"]]
    print(sol.canMakeSquare(grid1))  # Expected output: True

    grid2 = [["B","W","B"], ["W","B","W"], ["B","W","B"]]
    print(sol.canMakeSquare(grid2))  # Expected output: False

    grid3 = [["B","W","B"], ["B","W","W"], ["B","W","W"]]
    print(sol.canMakeSquare(grid3))  # Expected output: True