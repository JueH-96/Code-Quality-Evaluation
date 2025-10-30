class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        def check_sequence(val, pos):
            if pos == 0:
                return val == 1
            return (pos % 2 == 1 and val == 2) or (pos % 2 == 0 and val == 0)
        
        def explore(i, j, di, dj, pos, turned):
            if not (0 <= i < n and 0 <= j < m):
                return 0
            
            if not check_sequence(grid[i][j], pos):
                return 0
            
            # Continue in same direction
            length = 1 + explore(i + di, j + dj, di, dj, pos + 1, turned)
            
            # Try turning if not turned yet
            if not turned:
                # Try all possible 90-degree clockwise turns
                if (di, dj) == (1, 1):  # going down-right
                    length = max(length, 1 + explore(i + 1, j - 1, 1, -1, pos + 1, True))
                elif (di, dj) == (1, -1):  # going down-left
                    length = max(length, 1 + explore(i - 1, j - 1, -1, -1, pos + 1, True))
                elif (di, dj) == (-1, -1):  # going up-left
                    length = max(length, 1 + explore(i - 1, j + 1, -1, 1, pos + 1, True))
                elif (di, dj) == (-1, 1):  # going up-right
                    length = max(length, 1 + explore(i + 1, j + 1, 1, 1, pos + 1, True))
                    
            return length
        
        max_length = 0
        # Try all possible starting positions and directions
        for i in range(n):
            for j in range(m):
                # Try all four diagonal directions
                directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
                for di, dj in directions:
                    max_length = max(max_length, explore(i, j, di, dj, 0, False))
                    
        return max_length