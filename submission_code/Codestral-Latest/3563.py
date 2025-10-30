class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from itertools import combinations

        rows = len(grid)
        cols = len(grid[0])

        # Generate all possible combinations of columns
        all_combinations = []
        for r in range(1, rows + 1):
            all_combinations.extend(combinations(range(cols), r))

        max_score = 0

        for combo in all_combinations:
            selected_values = set()
            current_score = 0
            for row in range(rows):
                col = combo[row % len(combo)]
                value = grid[row][col]
                if value in selected_values:
                    break
                selected_values.add(value)
                current_score += value
            else:
                max_score = max(max_score, current_score)

        return max_score