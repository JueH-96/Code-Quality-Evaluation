def main():
    beats = {'R': 'S', 'S': 'P', 'P': 'R'}
    
    N = int(input())
    S = input().strip()
    
    valid_moves = []
    for a in S:
        moves = []
        for m in ['R', 'P', 'S']:
            if m == a or beats[m] == a:
                moves.append(m)
        valid_moves.append(moves)
    
    # Initialize DP
    current_dp = {'R': -float('inf'), 'P': -float('inf'), 'S': -float('inf')}
    
    # Process the first character
    if len(valid_moves[0]) == 0:
        print(0)
        return
    
    for m in valid_moves[0]:
        if beats[m] == S[0]:
            current_dp[m] = 1
        else:
            current_dp[m] = 0
    
    for i in range(1, N):
        new_dp = {'R': -float('inf'), 'P': -float('inf'), 'S': -float('inf')}
        for m_current in valid_moves[i]:
            max_prev = -float('inf')
            for m_prev in ['R', 'P', 'S']:
                if m_prev != m_current and current_dp[m_prev] > max_prev:
                    max_prev = current_dp[m_prev]
            if max_prev == -float('inf'):
                continue  # No valid transition to this move
            a = S[i]
            contribution = 1 if beats[m_current] == a else 0
            new_dp[m_current] = max_prev + contribution
        current_dp = new_dp.copy()
    
    max_wins = max(current_dp.values())
    print(max_wins if max_wins != -float('inf') else 0)

if __name__ == '__main__':
    main()