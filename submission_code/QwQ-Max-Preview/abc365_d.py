def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    if N == 0:
        print(0)
        return
    
    # Initialize for the first character
    first_char = S[0]
    prev_tie = 0  # tie gives 0 wins
    prev_win = 1  # win gives 1 win
    
    for i in range(1, N):
        current_char = S[i]
        current_m_tie = current_char
        current_m_win = beats[current_char]
        
        # Previous moves' actual characters
        prev_tie_move = S[i-1]
        prev_win_move = beats[S[i-1]]
        
        # Calculate current_tie
        possible_tie = []
        if prev_tie_move != current_m_tie:
            possible_tie.append(prev_tie)
        if prev_win_move != current_m_tie:
            possible_tie.append(prev_win)
        current_tie = max(possible_tie) if possible_tie else -float('inf')
        
        # Calculate current_win
        possible_win = []
        if prev_tie_move != current_m_win:
            possible_win.append(prev_tie)
        if prev_win_move != current_m_win:
            possible_win.append(prev_win)
        current_win = (max(possible_win) + 1) if possible_win else -float('inf')
        
        # Update previous values for next iteration
        prev_tie, prev_win = current_tie, current_win
    
    print(max(prev_tie, prev_win))

if __name__ == '__main__':
    main()