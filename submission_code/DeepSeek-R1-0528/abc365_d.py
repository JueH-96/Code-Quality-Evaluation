def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    wins_dict = {'R': 1, 'P': 2, 'S': 0}
    allowed_dict = {
        'R': [0, 1],
        'P': [1, 2],
        'S': [0, 2]
    }
    
    dp_prev = [-10**9] * 3
    
    first_char = s[0]
    allowed_moves = allowed_dict[first_char]
    win_idx = wins_dict[first_char]
    for move in allowed_moves:
        if move == win_idx:
            dp_prev[move] = 1
        else:
            dp_prev[move] = 0
            
    for i in range(1, n):
        a_char = s[i]
        allowed_moves = allowed_dict[a_char]
        win_idx = wins_dict[a_char]
        dp_curr = [-10**9] * 3
        
        for move in allowed_moves:
            k1 = (move + 1) % 3
            k2 = (move + 2) % 3
            candidate = max(dp_prev[k1], dp_prev[k2])
            win_value = 1 if move == win_idx else 0
            dp_curr[move] = win_value + candidate
            
        dp_prev = dp_curr
        
    print(max(dp_prev))

if __name__ == '__main__':
    main()