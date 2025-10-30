class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check all 4 possible 2x2 squares
        for i in range(2):
            for j in range(2):
                # Count B in this 2x2 square
                count_b = 0
                
                for di in range(2):
                    for dj in range(2):
                        if grid[i + di][j + dj] == 'B':
                            count_b += 1
                
                # If count_b is not 2, we can make a uniform square with at most 1 change
                # count_b = 0 or 4: already uniform
                # count_b = 1 or 3: need exactly 1 change
                # count_b = 2: need 2 changes (not allowed)
                if count_b != 2:
                    return True
        
        return False