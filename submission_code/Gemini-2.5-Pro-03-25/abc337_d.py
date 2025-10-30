import sys

def solve():
    # Read input dimensions H, W, K
    H, W, K = map(int, sys.stdin.readline().split())
    
    # Read the grid into a list of strings
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize minimum cost to infinity. This will store the minimum number of '.' changes needed.
    min_cost = float('inf')

    # --- Horizontal Check ---

    # Calculate prefix sums for each row to efficiently count '.' and 'x' in windows.
    # dot_count_row[r][c] stores the count of '.' characters in row r from column 0 up to c-1.
    # x_count_row[r][c] stores the count of 'x' characters in row r from column 0 up to c-1.
    dot_count_row = [[0] * (W + 1) for _ in range(H)]
    x_count_row = [[0] * (W + 1) for _ in range(H)]

    for r in range(H):
        for c in range(W):
            # Update prefix sums based on the character at S[r][c]
            dot_count_row[r][c+1] = dot_count_row[r][c]
            x_count_row[r][c+1] = x_count_row[r][c]
            if S[r][c] == '.':
                dot_count_row[r][c+1] += 1
            elif S[r][c] == 'x':
                x_count_row[r][c+1] += 1

    # Check all possible horizontal windows of length K.
    # A horizontal window is possible only if the grid width W is at least K.
    if W >= K:
      for r in range(H): # Iterate through each row
          # Iterate through all possible starting columns `c` for a window of length K
          # The window starting at column `c` covers columns `c` to `c+K-1`.
          for c in range(W - K + 1): 
              # Calculate the number of 'x' in the window [c, c+K-1] using prefix sums.
              # The sum over range [start, end] is P[end+1] - P[start].
              # Here start=c, end=c+K-1. Indices for prefix sum array are c+K and c.
              num_x = x_count_row[r][c+K] - x_count_row[r][c]
              
              # If there are no 'x' characters in the window, this window could potentially become all 'o's.
              if num_x == 0:
                  # Calculate the number of '.' characters in the window. This is the cost to make this window all 'o's.
                  num_dots = dot_count_row[r][c+K] - dot_count_row[r][c]
                  # Update the overall minimum cost if this window's cost is lower.
                  min_cost = min(min_cost, num_dots)

    # --- Vertical Check ---

    # Calculate prefix sums for each column.
    # dot_count_col[r][c] stores the count of '.' characters in column c from row 0 up to r-1.
    # x_count_col[r][c] stores the count of 'x' characters in column c from row 0 up to r-1.
    dot_count_col = [[0] * W for _ in range(H + 1)]
    x_count_col = [[0] * W for _ in range(H + 1)]

    for c in range(W): # Iterate through each column
        for r in range(H): # Iterate through each row
            # Update prefix sums based on the character at S[r][c]
            dot_count_col[r+1][c] = dot_count_col[r][c]
            x_count_col[r+1][c] = x_count_col[r][c]
            if S[r][c] == '.':
                dot_count_col[r+1][c] += 1
            elif S[r][c] == 'x':
                x_count_col[r+1][c] += 1

    # Check all possible vertical windows of length K.
    # A vertical window is possible only if the grid height H is at least K.
    if H >= K:
      for c in range(W): # Iterate through each column
          # Iterate through all possible starting rows `r` for a window of length K
          # The window starting at row `r` covers rows `r` to `r+K-1`.
          for r in range(H - K + 1):
              # Calculate the number of 'x' in the window [r, r+K-1] using prefix sums.
              # Indices for prefix sum array are r+K and r.
              num_x = x_count_col[r+K][c] - x_count_col[r][c]
              
              # If there are no 'x' characters in the window, this window could potentially become all 'o's.
              if num_x == 0:
                  # Calculate the number of '.' characters in the window. This is the cost.
                  num_dots = dot_count_col[r+K][c] - dot_count_col[r][c]
                  # Update the overall minimum cost if this window's cost is lower.
                  min_cost = min(min_cost, num_dots)

    # After checking all horizontal and vertical windows:
    # If min_cost is still infinity, it means no valid window (without 'x') was found.
    if min_cost == float('inf'):
        print("-1")
    else:
        # Otherwise, print the minimum cost found.
        print(min_cost)

# Call the solve function to execute the logic
solve()