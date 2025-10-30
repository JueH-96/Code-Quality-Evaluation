# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    """
    Reads N strings, converts them to vertical writing with '*' padding,
    and prints the resulting M strings according to the problem specification.
    """
    # Read N from stdin: the number of input strings.
    n = int(sys.stdin.readline())
    
    # Read N strings from stdin.
    # .strip() removes leading/trailing whitespace, including the newline character
    # from each line read by sys.stdin.readline().
    s = [sys.stdin.readline().strip() for _ in range(n)]

    # Constraints state N >= 1 and the length of each string |S_i| >= 1.
    
    # Calculate M, the maximum length among the input strings S_i.
    max_len = 0
    # Create a list of lengths of the input strings.
    lengths = [len(si) for si in s]
    # Find the maximum length. Since N >= 1 and |S_i| >= 1 are guaranteed by constraints,
    # the lengths list is non-empty and contains positive integers. 
    # Therefore, max_len will be at least 1.
    if lengths: # This check is technically redundant due to constraints but is safe programming practice.
        max_len = max(lengths)

    # If max_len were 0 (which shouldn't happen under the given constraints, 
    # as it would imply either N=0 or all |S_i|=0), we would print nothing and exit.
    if max_len == 0:
        return 

    # Initialize a list of lists called `temp_t`. 
    # Each inner list `temp_t[j]` will store the characters for the j-th output string 
    # (T_{j+1} in 1-based indexing) before final processing (like removing trailing '*').
    # There will be M output strings in total, corresponding to the maximum length M.
    temp_t = [[] for _ in range(max_len)]

    # Populate the `temp_t` lists based on the vertical writing transformation rule.
    # Iterate through column indices `j` from 0 to M-1. Each `j` corresponds to the character index 
    # within the original strings S_i, and determines which output string T_{j+1} we are building.
    for j in range(max_len): 
        # Iterate through the input string indices `i` in reverse order: N-1 down to 0.
        # In the problem's 1-based indexing, this corresponds to processing S_N, S_{N-1}, ..., S_1.
        # The character order within each output string T_j is determined by this reverse iteration.
        # The character at position `k` (0-based) of the final output string T_{j+1} 
        # will originate from the input string S_{N-k}.
        for i in range(n - 1, -1, -1): 
            # Get the current input string `s[i]` (which corresponds to S_{i+1} in 1-based indexing).
            current_s = s[i] 
            
            # Check if the current column index `j` is within the bounds of `current_s`'s length.
            # Python uses 0-based indexing, so valid indices are 0 to len(current_s) - 1.
            if j < len(current_s):
                # If `j` is a valid index for `current_s`, append the character `s[i][j]` 
                # to the list `temp_t[j]`. This character becomes part of the output string T_{j+1}.
                temp_t[j].append(current_s[j])
            else:
                # If `j` is out of bounds (i.e., `j >= len(current_s)`), this position 
                # should be filled with a placeholder character '*' according to the problem description.
                # This handles padding for strings shorter than the maximum length M.
                temp_t[j].append('*')

    # Process the temporary lists in `temp_t` to form the final output strings T_j and print them.
    # Iterate through the column indices again, from 0 to M-1, corresponding to T_1 to T_M.
    for j in range(max_len):
        # Join the characters collected in `temp_t[j]` into a single string. 
        # This forms the raw version of the output string T_{j+1}.
        raw_tj = "".join(temp_t[j])
        
        # Remove any trailing '*' characters from the `raw_tj` string. This is required by the condition:
        # "Each T_i does not end with *." The `rstrip('*')` method efficiently does this.
        final_tj = raw_tj.rstrip('*')
        
        # Print the resulting final string T_{j+1} to standard output.
        # The `print()` function automatically adds a newline character at the end.
        print(final_tj)

# Execute the solve function when the script is run.
solve()