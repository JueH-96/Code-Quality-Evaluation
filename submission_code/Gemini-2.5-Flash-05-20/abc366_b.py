import sys

def solve():
    N = int(sys.stdin.readline())
    
    strings = []
    max_len = 0
    for _ in range(N):
        s = sys.stdin.readline().strip()
        strings.append(s)
        # Keep track of the maximum length among all input strings
        if len(s) > max_len:
            max_len = len(s)
            
    M = max_len # M is the maximum length of the input strings, which determines the number of output strings

    # Initialize the grid (matrix) that will hold the characters for vertical writing.
    # The grid will have M rows (corresponding to T_1 to T_M) and N columns (corresponding to the transformation of S_1 to S_N).
    # All cells are initially filled with an asterisk '*' to represent empty spaces.
    grid = [['*' for _ in range(N)] for _ in range(M)]

    # Populate the grid based on the problem's transformation rule.
    # For each input string S_i (0-indexed as strings[i]):
    # The characters of S_i contribute to a specific column in the grid.
    # The problem specifies that S_i (1-indexed from 1 to N) relates to the (N-i+1)-th character of T_j.
    # Translating to 0-indexed:
    # If `i` is the 0-indexed position of the input string (from 0 to N-1),
    # then this string corresponds to column `N - 1 - i` in our 0-indexed grid.
    # The character at `strings[i][j]` (where `j` is the 0-indexed character position within the string)
    # belongs to the `j`-th row of the grid.
    for i in range(N): # Iterate through each input string S_i (0-indexed by `i`)
        current_s = strings[i]
        # Calculate the target column for the characters of the current string S_i.
        # S_0 goes to column N-1, S_1 to N-2, ..., S_{N-1} to column 0.
        target_col = N - 1 - i 
        
        # Place each character of the current string into the calculated grid column.
        for j in range(len(current_s)): # Iterate through characters of S_i (0-indexed by `j`)
            grid[j][target_col] = current_s[j]
            
    # After populating the entire grid, construct and print the output strings T_j.
    # Each T_j is formed by concatenating the characters in the `j`-th row of the grid.
    for j in range(M): # Iterate through each row of the grid (representing T_j, 0-indexed by `j`)
        t_j_list = grid[j] # Get the list of characters forming the current T_j string.
        t_j_str = "".join(t_j_list) # Join the characters to form the string.
        
        # Apply the crucial condition: "Each T_i does not end with *"
        # Use rstrip('*') to remove any trailing asterisks from the string.
        trimmed_t_j_str = t_j_str.rstrip('*')
        
        # Print the resulting T_j string.
        print(trimmed_t_j_str)

solve()