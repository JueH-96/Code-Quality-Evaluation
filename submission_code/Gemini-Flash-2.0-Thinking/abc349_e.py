class Solution:
    def solve(self, initial_values):
        memo = {}

        def check_win(grid, color):
            # Check rows
            for row in grid:
                if all(c == color for c in row):
                    return True
            # Check columns
            for j in range(3):
                if all(grid[i][j] == color for i in range(3)):
                    return True
            # Check diagonals
            if all(grid[i][i] == color for i in range(3)):
                return True
            if all(grid[i][2 - i] == color for i in range(3)):
                return True
            return False

        def get_white_cells(grid):
            white_cells = []
            for i in range(3):
                for j in range(3):
                    if grid[i][j] == 'W':
                        white_cells.append((i, j))
            return white_cells

        def calculate_scores(grid, initial_values):
            takahashi_score = 0
            aoki_score = 0
            for i in range(3):
                for j in range(3):
                    if grid[i][j] == 'R':
                        takahashi_score += initial_values[i][j]
                    elif grid[i][j] == 'B':
                        aoki_score += initial_values[i][j]
            return takahashi_score, aoki_score

        def find_winner(grid, turn):
            grid_tuple = tuple(map(tuple, grid))
            if grid_tuple in memo:
                return memo[grid_tuple]

            # Base cases
            if check_win(grid, 'R'):
                return 1
            if check_win(grid, 'B'):
                return -1
            if not get_white_cells(grid):
                t_score, a_score = calculate_scores(grid, initial_values)
                return 1 if t_score > a_score else -1

            white_cells = get_white_cells(grid)

            if turn == 0:  # Takahashi
                can_win = False
                for r, c in white_cells:
                    new_grid = list(map(list, grid))
                    new_grid[r][c] = 'R'
                    if find_winner(tuple(map(tuple, new_grid)), 1) == 1:
                        can_win = True
                        break
                memo[grid_tuple] = 1 if can_win else -1
                return memo[grid_tuple]
            else:  # Aoki
                can_win = False
                for r, c in white_cells:
                    new_grid = list(map(list, grid))
                    new_grid[r][c] = 'B'
                    if find_winner(tuple(map(tuple, new_grid)), 0) == -1:
                        can_win = True
                        break
                memo[grid_tuple] = -1 if can_win else 1
                return memo[grid_tuple]

        initial_grid_state = (('W', 'W', 'W'), ('W', 'W', 'W'), ('W', 'W', 'W'))
        winner = find_winner(initial_grid_state, 0)

        return "Takahashi" if winner == 1 else "Aoki"

initial_values_input = []
for _ in range(3):
    initial_values_input.append(list(map(int, input().split())))

solver = Solution()
result = solver.solve(initial_values_input)
print(result)