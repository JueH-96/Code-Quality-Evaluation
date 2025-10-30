class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Initialize counts for the colors
        bw, bw_cnt, wb, wb_cnt = set(), 0, set(), 0
        # Count the number of BW and WB occurrences on the 4 sides of each potential 2x2 square
        for i in range(3):
            for j in range(3):
                if j + 1 < 3:
                    bw.add((i, j))
                    bw_cnt += grid[i][j] == 'B' and grid[i][j + 1] == 'W'
                    wb.add((i, j))
                    wb_cnt += grid[i][j] == 'W' and grid[i][j + 1] == 'B'
                if i + 1 < 3:
                    bw.add((i, j))
                    bw_cnt += grid[i][j] == 'B' and grid[i + 1][j] == 'W'
                    wb.add((i, j))
                    wb_cnt += grid[i][j] == 'W' and grid[i + 1][j] == 'B'
        
        # Check if there already exists a 2x2 monochromatic square
        if bw_cnt <= len(bw) - 1 and wb_cnt <= len(wb) - 1:
            return True

        # Check if by changing a single cell we can fix all BW WB transitions
        for i, j in bw.union(wb):
            fix, nfix = 0, 0
            if grid[i][j] == 'B' and grid[i][j+1] == 'W' or grid[i][j] == 'W' and grid[i + 1][j] == 'B':
                fix += 1
            if grid[i][j + 1] == 'B' and grid[i + 1][j + 1] == 'W' or grid[i + 1][j] == 'B' and grid[i + 1][j+1] == 'W':
                fix += 1
                
            if grid[i][j] == 'W' and grid[i][j+1] == 'B' or grid[i][j] == 'B' and grid[i + 1][j] == 'W':
                nfix += 1
            if grid[i][j + 1] == 'W' and grid[i + 1][j + 1] == 'B' or grid[i + 1][j] == 'W' and grid[i + 1][j+1] == 'B':
                nfix += 1
                
            if fix != 0 and bw_cnt - fix + 1 <= len(bw) - 1 and wb_cnt - nfix <= len(wb) - 1:
                return True
        return False