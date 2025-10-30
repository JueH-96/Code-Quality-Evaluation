# YOUR CODE HERE
def check_win(grid, color):
    for i in range(3):
        if all(grid[i][j] == color for j in range(3)) or all(grid[j][i] == color for j in range(3)):
            return True
    if all(grid[i][i] == color for i in range(3)) or all(grid[i][2-i] == color for i in range(3)):
        return True
    return False

def minimax(grid, scores, is_takahashi, alpha, beta):
    if check_win(grid, 'R'):
        return 1
    if check_win(grid, 'B'):
        return -1
    
    if all(grid[i][j] != 'W' for i in range(3) for j in range(3)):
        return 1 if sum(scores) > 0 else -1

    best_score = float('-inf') if is_takahashi else float('inf')
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 'W':
                grid[i][j] = 'R' if is_takahashi else 'B'
                new_scores = scores + [grid[i][j]] if is_takahashi else scores - [grid[i][j]]
                score = minimax(grid, new_scores, not is_takahashi, alpha, beta)
                grid[i][j] = 'W'
                
                if is_takahashi:
                    best_score = max(best_score, score)
                    alpha = max(alpha, best_score)
                else:
                    best_score = min(best_score, score)
                    beta = min(beta, best_score)
                
                if beta <= alpha:
                    break
        if beta <= alpha:
            break
    
    return best_score

grid = [['W' for _ in range(3)] for _ in range(3)]
scores = []

for i in range(3):
    row = list(map(int, input().split()))
    for j in range(3):
        scores.append(row[j])

result = minimax(grid, [], True, float('-inf'), float('inf'))
print("Takahashi" if result > 0 else "Aoki")