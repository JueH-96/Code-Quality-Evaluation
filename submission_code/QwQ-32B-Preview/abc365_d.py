def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Define moves
    R, P, S = 0, 1, 2
    moves = ['R', 'P', 'S']
    
    # Initialize previous DP values
    if S[0] == 'R':
        prev_R = 0  # draw
        prev_P = 1  # win
        prev_S = float('-inf')  # invalid
    elif S[0] == 'P':
        prev_R = 1  # win
        prev_P = 0  # draw
        prev_S = float('-inf')  # invalid
    else:  # S[0] == 'S'
        prev_R = 0  # draw
        prev_P = float('-inf')  # invalid
        prev_S = 1  # win
    
    for i in range(1, N):
        s = S[i]
        if s == 'R':
            # Takahashi can choose P (win) or R (draw)
            current_P = max(prev_R, prev_S) + 1  # win
            current_R = max(prev_P, prev_S) + 0  # draw
            current_S = float('-inf')
        elif s == 'P':
            # Takahashi can choose S (win) or P (draw)
            current_S = max(prev_R, prev_S) + 1  # win
            current_P = max(prev_R, prev_S) + 0  # draw
            current_R = float('-inf')
        else:  # s == 'S'
            # Takahashi can choose R (win) or S (draw)
            current_R = max(prev_P, prev_S) + 1  # win
            current_S = max(prev_R, prev_P) + 0  # draw
            current_P = float('-inf')
        
        # Update previous values for next iteration
        prev_R, prev_P, prev_S = current_R, current_P, current_S
    
    # The answer is the maximum of the last moves
    print(max(prev_R, prev_P, prev_S))

if __name__ == '__main__':
    main()