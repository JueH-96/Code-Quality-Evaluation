def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    allowed_moves = []
    for c in S:
        if c == 'R':
            allowed_moves.append(['R', 'P'])
        elif c == 'P':
            allowed_moves.append(['P', 'S'])
        else:
            allowed_moves.append(['S', 'R'])
    
    if N == 0:
        print(0)
        return
    
    # Initialize dp_prev
    dp_prev = {}
    for m in allowed_moves[0]:
        if m == S[0]:
            dp_prev[m] = 0
        else:
            dp_prev[m] = 1
    
    for i in range(1, N):
        current_dp = {}
        current_allowed = allowed_moves[i]
        prev_allowed = allowed_moves[i-1]
        aoki_move = S[i]
        for current_move in current_allowed:
            # Find candidates in previous allowed moves that are not equal to current_move
            candidates = []
            for pm in prev_allowed:
                if pm != current_move:
                    candidates.append(pm)
            # Get max_prev value
            if len(candidates) == 1:
                max_prev = dp_prev[candidates[0]]
            else:
                max_prev = max(dp_prev[candidates[0]], dp_prev[candidates[1]])
            # Determine current_win
            if current_move == aoki_move:
                current_win = 0
            else:
                current_win = 1
            current_dp[current_move] = max_prev + current_win
        dp_prev = current_dp
    
    print(max(dp_prev.values()))

if __name__ == "__main__":
    main()