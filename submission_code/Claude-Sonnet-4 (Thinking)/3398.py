class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all possible 2x2 squares
        for i in range(2):
            for j in range(2):
                # Collect colors in the 2x2 square starting at (i, j)
                colors = []
                for di in range(2):
                    for dj in range(2):
                        colors.append(grid[i + di][j + dj])
                
                b_count = colors.count('B')
                w_count = colors.count('W')
                
                # If we can make all cells the same color by changing at most 1 cell
                if b_count >= 3 or w_count >= 3:
                    return True
        
        return False