class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        max_score = 0

        for i in range(1 << (rows * cols)):
            selected_cells = []
            for r in range(rows):
                for c in range(cols):
                    if (i >> (r * cols + c)) & 1:
                        selected_cells.append((r, c))
            
            if not selected_cells:
                continue

            valid_selection = True
            selected_values = set()
            selected_rows = set()

            for r, c in selected_cells:
                if r in selected_rows:
                    valid_selection = False
                    break
                selected_rows.add(r)
                val = grid[r][c]
                if val in selected_values:
                    valid_selection = False
                    break
                selected_values.add(val)
            
            if valid_selection:
                current_score = sum(grid[r][c] for r, c in selected_cells)
                max_score = max(max_score, current_score)
        
        return max_score