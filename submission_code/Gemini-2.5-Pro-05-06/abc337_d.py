import sys

def solve():
    H, W, K = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    min_ops = float('inf')

    # Horizontal pass
    # Only proceed if K is not greater than the width W,
    # meaning horizontal K-length sequences are possible.
    if K <= W:
        for r in range(H):
            current_dots = 0
            current_xs = 0
            
            # Initialize counts for the first window S[r][0...K-1]
            for c_idx in range(K):
                char = S[r][c_idx]
                if char == '.':
                    current_dots += 1
                elif char == 'x':
                    current_xs += 1
            
            # Check the first window
            if current_xs == 0:
                min_ops = min(min_ops, current_dots)
            
            # Slide the window across the row
            # c_start_idx is the starting column index of the NEW window.
            # New window is S[r][c_start_idx ... c_start_idx + K - 1].
            # Loop runs for c_start_idx from 1 to W-K.
            for c_start_idx in range(1, W - K + 1):
                # Character leaving the window (S[r][c_start_idx - 1])
                char_out = S[r][c_start_idx - 1]
                if char_out == '.':
                    current_dots -= 1
                elif char_out == 'x':
                    current_xs -= 1
                
                # Character entering the window (S[r][c_start_idx + K - 1])
                char_in = S[r][c_start_idx + K - 1]
                if char_in == '.':
                    current_dots += 1
                elif char_in == 'x':
                    current_xs += 1
                
                # Check the current window
                if current_xs == 0:
                    min_ops = min(min_ops, current_dots)

    # Vertical pass
    # Only proceed if K is not greater than the height H,
    # meaning vertical K-length sequences are possible.
    if K <= H:
        for c in range(W):
            current_dots = 0
            current_xs = 0

            # Initialize counts for the first window S[0...K-1][c]
            for r_idx in range(K):
                char = S[r_idx][c]
                if char == '.':
                    current_dots += 1
                elif char == 'x':
                    current_xs += 1
            
            # Check the first window
            if current_xs == 0:
                min_ops = min(min_ops, current_dots)

            # Slide the window down the column
            # r_start_idx is the starting row index of the NEW window.
            # New window is S[r_start_idx ... r_start_idx + K - 1][c].
            # Loop runs for r_start_idx from 1 to H-K.
            for r_start_idx in range(1, H - K + 1):
                # Character leaving the window (S[r_start_idx - 1][c])
                char_out = S[r_start_idx - 1][c]
                if char_out == '.':
                    current_dots -= 1
                elif char_out == 'x':
                    current_xs -= 1
                
                # Character entering the window (S[r_start_idx + K - 1][c])
                char_in = S[r_start_idx + K - 1][c]
                if char_in == '.':
                    current_dots += 1
                elif char_in == 'x':
                    current_xs += 1
                
                # Check the current window
                if current_xs == 0:
                    min_ops = min(min_ops, current_dots)

    if min_ops == float('inf'):
        print("-1")
    else:
        print(min_ops)

solve()