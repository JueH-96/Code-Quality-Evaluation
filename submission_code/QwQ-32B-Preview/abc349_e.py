import sys
sys.setrecursionlimit(100000)

values = []
colors = [['W', 'W', 'W'], ['W', 'W', 'W'], ['W', 'W', 'W']]

# Define winning lines
winning_lines = [
    [(0,0), (0,1), (0,2)],  # row 1
    [(1,0), (1,1), (1,2)],  # row 2
    [(2,0), (2,1), (2,2)],  # row 3
    [(0,0), (1,0), (2,0)],  # column 1
    [(0,1), (1,1), (2,1)],  # column 2
    [(0,2), (1,2), (2,2)],  # column 3
    [(0,0), (1,1), (2,2)],  # diagonal top-left to bottom-right
    [(0,2), (1,1), (2,0)],  # diagonal top-right to bottom-left
]

def read_grid():
    global values
    values = [list(map(int, input().split())) for _ in range(3)]

def check_win(colors, color):
    for line in winning_lines:
        if all(colors[r][c] == color for (r, c) in line):
            return True
    return False

def is_grid_full(colors):
    for row in colors:
        if 'W' in row:
            return False
    return True

def get_possible_moves(colors):
    moves = []
    for i in range(3):
        for j in range(3):
            if colors[i][j] == 'W':
                moves.append((i, j))
    return moves

def make_move(colors, move, color):
    new_colors = [list(row) for row in colors]
    new_colors[move[0]][move[1]] = color
    return new_colors

def minimax(grid, colors, scores, current_player, alpha, beta):
    if check_win(colors, 'R'):
        return 1
    elif check_win(colors, 'B'):
        return -1
    elif is_grid_full(colors):
        return 1 if scores['T'] > scores['A'] else -1
    else:
        if current_player == 'T':
            best_val = -float('inf')
            for move in get_possible_moves(colors):
                new_colors = make_move(colors, move, 'R')
                new_scores = scores.copy()
                new_scores['T'] += values[move[0]][move[1]]
                val = minimax(grid, new_colors, new_scores, 'A', alpha, beta)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val
        else:  # current_player == 'A'
            best_val = float('inf')
            for move in get_possible_moves(colors):
                new_colors = make_move(colors, move, 'B')
                new_scores = scores.copy()
                new_scores['A'] += values[move[0]][move[1]]
                val = minimax(grid, new_colors, new_scores, 'T', alpha, beta)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val

def main():
    read_grid()
    scores = {'T': 0, 'A': 0}
    result = minimax(values, colors, scores, 'T', -float('inf'), float('inf'))
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()