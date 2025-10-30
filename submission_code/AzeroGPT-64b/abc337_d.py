def solve(H, W, K, S):
    row_scores = []
    col_scores = []
    for s in S:
        score = 0
        last_o = -1
        for i, ch in enumerate(s):
            if ch == 'o':
                if last_o >= 0:
                    score = max(score, i - last_o + 1)
                    score = score if score <= K else K if i - last_o >= K else K + 1
                last_o = i
            elif (last_o >= 0):
                score = i - last_o
                score = score if score <= K else K if i - last_o >= K else K + 1

        row_scores.append(score)
    
    for i in range(W):
        score = 0
        last_o = -1
        for j in range(H):
            if S[j][i] == 'o':
                if last_o >= 0:
                    score = max(score, j - last_o + 1)
                    score = score if score <= K else K if j - last_o >= K else K + 1
                last_o = j
            elif (last_o >= 0):
                score = j - last_o
                score = score if score <= K else K if j - last_o >= K else K + 1
                    
        col_scores.append(score)
    min_ops = float('inf')
    for row in range(H):
        if row_scores[row] >= K:
            for check_row in range(max(row-K, 0), min(row + K, H)):
                min_ops = min(min_ops, max(0, K - row_scores[check_row]))
        elif row_scores[row] > 0:
            if row - 1 >= 0:
                min_ops = min(min_ops, max(0, K - row_scores[row]) + max(0, K - row_scores[row-1]))
            if row + 1 < H:
                min_ops = min(min_ops, max(0, K - row_scores[row]) + max(0, K - row_scores[row+1]))
                
    for col in range(W):
        if col_scores[col] >= K:
            for check_col in range(max(col-K, 0), min(col + K, W)):
                min_ops = min(min_ops, max(0, K - col_scores[check_col]))

        elif col_scores[col] > 0:
            if col - 1 >= 0:
                min_ops = min(min_ops,  max(0, K - col_scores[col]) + max(0, K - col_scores[col-1]))
            if col + 1 < W:
                min_ops = min(min_ops,  max(0, K - col_scores[col]) + max(0, K - col_scores[col+1]))
    if min_ops == float('inf') or min_ops < 0:
        min_ops = -1
        
    return min_ops

H, W, K = map(int, input().split())
S = [input() for _ in range(H)]
my_result = solve(H, W, K, S)
print(my_result)