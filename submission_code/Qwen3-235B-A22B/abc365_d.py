import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    allowed_moves = {
        'R': {'R', 'P'},
        'P': {'P', 'S'},
        'S': {'S', 'R'},
    }

    prev_r = float('-inf')
    prev_p = float('-inf')
    prev_s = float('-inf')

    for i in range(N):
        c = S[i]

        # Determine current win values
        if c == 'R':
            curr_win_r = 0
            curr_win_p = 1
            curr_win_s = 0
        elif c == 'P':
            curr_win_r = 0
            curr_win_p = 0
            curr_win_s = 1
        else:  # c == 'S'
            curr_win_r = 1
            curr_win_p = 0
            curr_win_s = 0

        allowed = allowed_moves[c]

        new_r = float('-inf')
        new_p = float('-inf')
        new_s = float('-inf')

        # Update for move R if allowed
        if 'R' in allowed:
            if i == 0:
                new_r = curr_win_r
            else:
                new_r = max(prev_p, prev_s) + curr_win_r

        # Update for move P if allowed
        if 'P' in allowed:
            if i == 0:
                new_p = curr_win_p
            else:
                new_p = max(prev_r, prev_s) + curr_win_p

        # Update for move S if allowed
        if 'S' in allowed:
            if i == 0:
                new_s = curr_win_s
            else:
                new_s = max(prev_r, prev_p) + curr_win_s

        # Update previous values for next iteration
        prev_r, prev_p, prev_s = new_r, new_p, new_s

    result = max(prev_r, prev_p, prev_s)
    print(result)

if __name__ == '__main__':
    main()