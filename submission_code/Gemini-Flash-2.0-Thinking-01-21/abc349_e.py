import collections

def solve():
    a_matrix = []
    for _ in range(3):
        a_matrix.append(list(map(int, input().split())))
    
    initial_grid = [[0] * 3 for _ in range(3)] # 0: white, 1: red, 2: blue
    memo = {}
    
    def get_winning_line_color(grid):
        # Check rows
        for i in range(3):
            if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != 0:
                return grid[i][0]
        # Check columns
        for j in range(3):
            if grid[0][j] == grid[1][j] == grid[2][j] and grid[0][j] != 0:
                return grid[0][j]
        # Check diagonals
        if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 0:
            return grid[0][0]
        if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 0:
            return grid[0][2]
        return 0
        
    def get_white_cells(grid):
        cells = []
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    cells.append((i, j))
        return cells
        
    def calculate_scores(grid):
        takahashi_score = 0
        aoki_score = 0
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 1:
                    takahashi_score += a_matrix[i][j]
                elif grid[i][j] == 2:
                    aoki_score += a_matrix[i][j]
        return takahashi_score, aoki_score
        
    def get_winner(grid, is_takahashi_turn):
        grid_tuple = tuple(tuple(row) for row in grid)
        state = (grid_tuple, is_takahashi_turn)
        if state in memo:
            return memo[state]
            
        winning_color = get_winning_line_color(grid)
        if winning_color == 1:
            return "Takahashi"
        elif winning_color == 2:
            return "Aoki"
            
        white_cells = get_white_cells(grid)
        if not white_cells:
            t_score, a_score = calculate_scores(grid)
            if t_score > a_score:
                return "Takahashi"
            else:
                return "Aoki"
                
        if is_takahashi_turn:
            takahashi_can_win = False
            for r, c in white_cells:
                next_grid = [list(row) for row in grid]
                next_grid[r][c] = 1
                outcome = get_winner(next_grid, False)
                if outcome == "Takahashi":
                    takahashi_can_win = True
                    break
            result = "Takahashi" if takahashi_can_win else "Aoki"
        else:
            aoki_can_win = False
            for r, c in white_cells:
                next_grid = [list(row) for row in grid]
                next_grid[r][c] = 2
                outcome = get_winner(next_grid, True)
                if outcome == "Aoki":
                    aoki_can_win = True
                    break
            result = "Aoki" if aoki_can_win else "Takahashi"
            
        memo[state] = result
        return result
        
    winner = get_winner(initial_grid, True)
    print(winner)

if __name__ == '__main__':
    solve()