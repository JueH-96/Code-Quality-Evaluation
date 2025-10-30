def get_options(aoki_move):
    if aoki_move == 'R':
        return ('P', 'R')  # P beats R, R ties R
    elif aoki_move == 'P':
        return ('S', 'P')  # S beats P, P ties P
    elif aoki_move == 'S':
        return ('R', 'S')  # R beats S, S ties S

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Initialize previous moves with -infinity
    prev_R = float('-inf')
    prev_P = float('-inf')
    prev_S = float('-inf')
    
    # First game
    win_move, tie_move = get_options(S[0])
    if win_move == 'R':
        prev_R = 1
    elif win_move == 'P':
        prev_P = 1
    elif win_move == 'S':
        prev_S = 1
    if tie_move == 'R':
        prev_R = max(prev_R, 0)
    elif tie_move == 'P':
        prev_P = max(prev_P, 0)
    elif tie_move == 'S':
        prev_S = max(prev_S, 0)
    
    # Iterate through the remaining games
    for i in range(1, N):
        win_move, tie_move = get_options(S[i])
        current_R = float('-inf')
        current_P = float('-inf')
        current_S = float('-inf')
        
        # Update for winning move
        if win_move == 'R':
            current_R = max(prev_P, prev_S) + 1
        elif win_move == 'P':
            current_P = max(prev_R, prev_S) + 1
        elif win_move == 'S':
            current_S = max(prev_R, prev_P) + 1
        
        # Update for tie move
        if tie_move == 'R':
            current_R = max(current_R, prev_P, prev_S)
        elif tie_move == 'P':
            current_P = max(current_P, prev_R, prev_S)
        elif tie_move == 'S':
            current_S = max(current_S, prev_R, prev_P)
        
        # Update previous moves for next iteration
        prev_R, prev_P, prev_S = current_R, current_P, current_S
    
    # The answer is the maximum of the previous moves
    answer = max(prev_R, prev_P, prev_S)
    print(answer)

if __name__ == '__main__':
    main()