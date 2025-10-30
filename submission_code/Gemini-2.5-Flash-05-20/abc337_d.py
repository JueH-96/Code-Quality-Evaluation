import sys

def solve():
    # Read H, W, K from the first line
    H, W, K = map(int, sys.stdin.readline().split())

    # Read the grid S. Each S_i is a string.
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize min_ops to a very large number (infinity)
    # This will store the minimum number of operations found.
    min_ops = float('inf')

    # --- Process horizontal segments ---
    # Iterate through each row
    for r in range(H):
        dot_count = 0  # Counts '.' characters in the current sliding window
        x_count = 0    # Counts 'x' characters in the current sliding window

        # Iterate through each column, treating 'c' as the rightmost character of the window
        for c in range(W):
            # Add the character at S[r][c] to the current window
            if S[r][c] == '.':
                dot_count += 1
            elif S[r][c] == 'x':
                x_count += 1

            # Check if the window has reached the size K.
            # The window starts at column 'c - K + 1' and ends at 'c' (inclusive).
            if c >= K - 1:
                # If there are no 'x' characters in the current window,
                # this segment is a valid candidate for K consecutive 'o's.
                # The number of operations needed is simply the count of '.'s.
                if x_count == 0:
                    min_ops = min(min_ops, dot_count)

                # Now, prepare the window for the next iteration by
                # removing the character that is leaving the window from the left side.
                left_char_idx = c - K + 1
                if S[r][left_char_idx] == '.':
                    dot_count -= 1
                elif S[r][left_char_idx] == 'x':
                    x_count -= 1

    # --- Process vertical segments ---
    # Iterate through each column
    for c in range(W):
        dot_count = 0  # Counts '.' characters in the current sliding window
        x_count = 0    # Counts 'x' characters in the current sliding window

        # Iterate through each row, treating 'r' as the bottommost character of the window
        for r in range(H):
            # Add the character at S[r][c] to the current window
            if S[r][c] == '.':
                dot_count += 1
            elif S[r][c] == 'x':
                x_count += 1

            # Check if the window has reached the size K.
            # The window starts at row 'r - K + 1' and ends at 'r' (inclusive).
            if r >= K - 1:
                # If there are no 'x' characters in the current window,
                # this segment is a valid candidate for K consecutive 'o's.
                # The number of operations needed is simply the count of '.'s.
                if x_count == 0:
                    min_ops = min(min_ops, dot_count)

                # Now, prepare the window for the next iteration by
                # removing the character that is leaving the window from the top side.
                top_char_idx = r - K + 1
                if S[top_char_idx][c] == '.':
                    dot_count -= 1
                elif S[top_char_idx][c] == 'x':
                    x_count -= 1

    # After checking all possible horizontal and vertical segments:
    # If min_ops is still infinity, it means no valid segment (i.e., one without 'x's) was found.
    if min_ops == float('inf'):
        print(-1)  # Impossible to satisfy the condition
    else:
        print(min_ops) # Print the minimum number of operations required

# Call the solve function to execute the program
solve()