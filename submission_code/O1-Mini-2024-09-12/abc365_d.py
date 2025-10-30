def main():
    import sys
    import sys
    N, S = sys.stdin.read().split()
    N = int(N)
    win_move = {'R':'P', 'P':'S', 'S':'R'}
    move_map = {'R':0, 'P':1, 'S':2}
    INF = -1 << 60
    dp_prev = [0, 0, 0]  # Without previous move constraint at start
    for i in range(N):
        a_move = S[i]
        win = win_move[a_move]
        draw = a_move
        dp_curr = [INF]*3
        for prev in range(3):
            # Choose to win
            current_move = win
            current_idx = move_map[current_move]
            if current_idx != prev:
                dp_curr[current_idx] = max(dp_curr[current_idx], dp_prev[prev] +1)
            # Choose to draw
            current_move = draw
            current_idx = move_map[current_move]
            if current_idx != prev:
                dp_curr[current_idx] = max(dp_curr[current_idx], dp_prev[prev])
        dp_prev = dp_curr
    print(max(dp_prev))
    
if __name__ == "__main__":
    main()