def read_input():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def check_consecutive(grid_colors):
    # Check rows
    for i in range(3):
        if grid_colors[i][0] != 0 and grid_colors[i][0] == grid_colors[i][1] == grid_colors[i][2]:
            return grid_colors[i][0]
    
    # Check columns
    for j in range(3):
        if grid_colors[0][j] != 0 and grid_colors[0][j] == grid_colors[1][j] == grid_colors[2][j]:
            return grid_colors[0][j]
    
    # Check diagonals
    if grid_colors[0][0] != 0 and grid_colors[0][0] == grid_colors[1][1] == grid_colors[2][2]:
        return grid_colors[0][0]
    if grid_colors[0][2] != 0 and grid_colors[0][2] == grid_colors[1][1] == grid_colors[2][0]:
        return grid_colors[0][2]
    
    return 0  # No consecutive cells of the same color

def is_game_over(grid_colors, takahashi_score, aoki_score):
    # Check if there are three consecutive cells of the same color
    winner = check_consecutive(grid_colors)
    if winner == 1:  # Takahashi
        return True, "Takahashi"
    elif winner == -1:  # Aoki
        return True, "Aoki"
    
    # Check if there are no white cells left
    white_cells = sum(row.count(0) for row in grid_colors)
    if white_cells == 0:
        if takahashi_score > aoki_score:
            return True, "Takahashi"
        else:
            return True, "Aoki"
    
    return False, ""

def minimax(grid, grid_colors, takahashi_score, aoki_score, is_takahashi, alpha, beta):
    game_over, winner = is_game_over(grid_colors, takahashi_score, aoki_score)
    if game_over:
        if winner == "Takahashi":
            return 1
        else:
            return -1
    
    if is_takahashi:
        best_val = float('-inf')
        for i in range(3):
            for j in range(3):
                if grid_colors[i][j] == 0:
                    grid_colors[i][j] = 1
                    val = minimax(grid, grid_colors, takahashi_score + grid[i][j], aoki_score, False, alpha, beta)
                    grid_colors[i][j] = 0
                    best_val = max(best_val, val)
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return best_val
    else:
        best_val = float('inf')
        for i in range(3):
            for j in range(3):
                if grid_colors[i][j] == 0:
                    grid_colors[i][j] = -1
                    val = minimax(grid, grid_colors, takahashi_score, aoki_score + grid[i][j], True, alpha, beta)
                    grid_colors[i][j] = 0
                    best_val = min(best_val, val)
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return best_val

def solve():
    grid = read_input()
    grid_colors = [[0 for _ in range(3)] for _ in range(3)]  # 0: white, 1: red (Takahashi), -1: blue (Aoki)
    takahashi_score = 0
    aoki_score = 0
    
    result = minimax(grid, grid_colors, takahashi_score, aoki_score, True, float('-inf'), float('inf'))
    
    if result == 1:
        return "Takahashi"
    else:
        return "Aoki"

print(solve())