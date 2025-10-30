import math

def solve():
    a_matrix = []
    for _ in range(3):
        a_matrix.append(list(map(int, input().split())))
    
    initial_grid = [['W'] * 3 for _ in range(3)]
    memo = {}
    
    def get_winner(grid):
        lines = []
        for i in range(3):
            lines.append([grid[i][0], grid[i][1], grid[i][2]]) # rows
            lines.append([grid[0][i], grid[1][i], grid[2][i]]) # columns
        lines.append([grid[0][0], grid[1][1], grid[2][2]]) # diagonal 1
        lines.append([grid[0][2], grid[1][1], grid[2][0]]) # diagonal 2
        for line in lines:
            if all(cell == 'R' for cell in line):
                return "Takahashi"
            if all(cell == 'B' for cell in line):
                return "Aoki"
        return "None"
        
    def has_white_cells(grid):
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 'W':
                    return True
        return False
        
    def get_game_result(current_grid, turn, score_takahashi, score_aoki):
        grid_tuple = tuple(tuple(row) for row in current_grid)
        state = (grid_tuple, turn)
        if state in memo:
            return memo[state]
            
        winner = get_winner(current_grid)
        if winner == "Takahashi":
            return 1
        if winner == "Aoki":
            return -1
            
        if not has_white_cells(current_grid):
            if score_takahashi > score_aoki:
                return 1
            else:
                return -1
                
        possible_moves = []
        for i in range(3):
            for j in range(3):
                if current_grid[i][j] == 'W':
                    possible_moves.append((i, j))
                    
        results = []
        for r, c in possible_moves:
            next_grid = [list(row) for row in current_grid]
            if turn == 'T':
                next_grid[r][c] = 'R'
                next_score_t = score_takahashi + a_matrix[r][c]
                next_score_a = score_aoki
                next_turn = 'A'
            else:
                next_grid[r][c] = 'B'
                next_score_t = score_takahashi
                next_score_a = score_aoki + a_matrix[r][c]
                next_turn = 'T'
            results.append(get_game_result(next_grid, next_turn, next_score_t, next_score_a))
            
        if turn == 'T':
            best_result = max(results)
        else:
            best_result = min(results)
            
        memo[state] = best_result
        return best_result
        
    final_result = get_game_result(initial_grid, 'T', 0, 0)
    if final_result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == '__main__':
    solve()