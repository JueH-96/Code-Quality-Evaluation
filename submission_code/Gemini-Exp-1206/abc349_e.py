def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))

    def check_win(board, player):
        for i in range(3):
            if all(board[i][j] == player for j in range(3)):
                return True
            if all(board[j][i] == player for j in range(3)):
                return True
        if all(board[i][i] == player for i in range(3)):
            return True
        if all(board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def calculate_score(board):
        takahashi_score = 0
        aoki_score = 0
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'R':
                    takahashi_score += grid[i][j]
                elif board[i][j] == 'B':
                    aoki_score += grid[i][j]
        return takahashi_score, aoki_score

    def minimax(board, depth, is_maximizing):
        if check_win(board, 'R'):
            return 1, 0
        if check_win(board, 'B'):
            return -1, 0
        
        moves = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    moves.append((i,j))
        
        if not moves:
            t_score, a_score = calculate_score(board)
            if t_score > a_score:
                return 1, 0
            else:
                return -1, 0

        if is_maximizing:
            best_score = -float('inf')
            best_move = None
            for r, c in moves:
                board[r][c] = 'R'
                score, _ = minimax(board, depth + 1, False)
                board[r][c] = 0
                if score > best_score:
                    best_score = score
                    best_move = (r,c)
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for r, c in moves:
                board[r][c] = 'B'
                score, _ = minimax(board, depth + 1, True)
                board[r][c] = 0
                if score < best_score:
                    best_score = score
                    best_move = (r,c)
            return best_score, best_move

    initial_board = [[0] * 3 for _ in range(3)]
    result, _ = minimax(initial_board, 0, True)

    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

solve()