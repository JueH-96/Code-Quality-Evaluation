# YOUR CODE HERE
N = int(input())
S = input().strip()

if N == 1:
    print(1)
else:
    moves = ['R', 'P', 'S']
    INF = -10**9
    dp = [[INF] * 3 for _ in range(N)]
    
    def can_play(takahashi_move, aoki_move):
        if takahashi_move == aoki_move:
            return True
        if (takahashi_move == 'R' and aoki_move == 'S') or \
           (takahashi_move == 'P' and aoki_move == 'R') or \
           (takahashi_move == 'S' and aoki_move == 'P'):
            return True
        return False
    
    def wins(takahashi_move, aoki_move):
        return (takahashi_move == 'R' and aoki_move == 'S') or \
               (takahashi_move == 'P' and aoki_move == 'R') or \
               (takahashi_move == 'S' and aoki_move == 'P')
    
    for j in range(3):
        if can_play(moves[j], S[0]):
            dp[0][j] = 1 if wins(moves[j], S[0]) else 0
    
    for i in range(1, N):
        for j in range(3):
            if can_play(moves[j], S[i]):
                for k in range(3):
                    if k != j and dp[i-1][k] != INF:
                        win_count = 1 if wins(moves[j], S[i]) else 0
                        dp[i][j] = max(dp[i][j], dp[i-1][k] + win_count)
    
    result = max(dp[N-1])
    print(result)