class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        max_score = 0
        
        for i in range(1 << (rows * cols)):
            selected_cells = []
            selected_rows = set()
            selected_values = set()
            current_score = 0
            valid = True
            
            for r in range(rows):
                for c in range(cols):
                    index = r * cols + c
                    if (i >> index) & 1:
                        if r in selected_rows:
                            valid = False
                            break
                        selected_rows.add(r)
                        selected_cells.append(grid[r][c])
                if not valid:
                    break
            
            if valid:
                if len(selected_cells) != len(set(selected_cells)):
                    continue
                current_score = sum(selected_cells)
                max_score = max(max_score, current_score)
        return max_score