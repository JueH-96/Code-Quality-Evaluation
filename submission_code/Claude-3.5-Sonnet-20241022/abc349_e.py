def check_win(grid, color):
    # Check rows
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] == color:
            return True
            
    # Check columns
    for j in range(3):
        if grid[0][j] == grid[1][j] == grid[2][j] == color:
            return True
            
    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] == color:
        return True
    if grid[0][2] == grid[1][1] == grid[2][0] == color:
        return True
        
    return False

def get_score(scores, moves):
    score = 0
    for i, j in moves:
        score += scores[i][j]
    return score

def minimax(scores, grid, is_takahashi, taka_moves, aoki_moves, alpha, beta):
    if check_win(grid, 'R'):
        return 1
    if check_win(grid, 'B'):
        return -1
        
    white_cells = []
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 'W':
                white_cells.append((i,j))
                
    if not white_cells:
        taka_score = get_score(scores, taka_moves)
        aoki_score = get_score(scores, aoki_moves)
        return 1 if taka_score > aoki_score else -1
        
    if is_takahashi:
        value = float('-inf')
        for i,j in white_cells:
            grid[i][j] = 'R'
            taka_moves.append((i,j))
            value = max(value, minimax(scores, grid, False, taka_moves, aoki_moves, alpha, beta))
            grid[i][j] = 'W'
            taka_moves.pop()
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for i,j in white_cells:
            grid[i][j] = 'B'
            aoki_moves.append((i,j))
            value = min(value, minimax(scores, grid, True, taka_moves, aoki_moves, alpha, beta))
            grid[i][j] = 'W'
            aoki_moves.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def solve():
    scores = []
    for _ in range(3):
        row = list(map(int, input().split()))
        scores.append(row)
        
    grid = [['W']*3 for _ in range(3)]
    result = minimax(scores, grid, True, [], [], float('-inf'), float('inf'))
    
    print("Takahashi" if result > 0 else "Aoki")

solve()