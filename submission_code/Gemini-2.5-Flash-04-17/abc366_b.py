# YOUR CODE HERE
import sys

# Read N, the number of strings
N = int(sys.stdin.readline())

# Read the N strings into a list S.
# Use strip() to remove the trailing newline character from each line read from stdin.
S = [sys.stdin.readline().strip() for _ in range(N)]

# Calculate M, the maximum length among the N input strings.
# This determines the number of output strings T_j.
M = 0
for s in S:
    M = max(M, len(s))

# Create a 2D list (matrix) to build the output strings.
# The matrix dimensions are M rows by N columns.
# Each row `j` (0-indexed, from 0 to M-1) will correspond to the output string T_{j+1}.
# Each column `k` (0-indexed, from 0 to N-1) will correspond to the (k+1)-th character position within T_{j+1}.
# Initialize all cells with the character '*' as specified in the problem.
output_matrix = [['*' for _ in range(N)] for _ in range(M)]

# Fill the matrix with characters from the input strings S_i.
# The problem defines the mapping (using 0-indexed notation for clarity in code):
# The character S[i][j] (j-th character of the i-th input string, 0 <= i < N, 0 <= j < |S[i]|)
# should be placed at the (N-1-i)-th position (0-indexed) within the string T[j] (0 <= j < M).
# In our matrix representation, this means S[i][j] goes into output_matrix[j][N - 1 - i].
# Loop through each input string S[i] by its index `i` (0 to N-1).
for i in range(N):
    # Loop through each character in the current input string S[i] by its index `j` (0 to len(S[i])-1).
    for j in range(len(S[i])):
        # Place the character S[i][j] into the designated cell in the matrix.
        # The row index is `j`, which corresponds to the output string T_{j+1}.
        # The column index is `N - 1 - i`, which corresponds to the position within T_{j+1}.
        output_matrix[j][N - 1 - i] = S[i][j]

# Generate and print the output strings T_j.
# We need to print M strings, one for each row of the output_matrix.
# Loop through each row index `j` (0 to M-1), corresponding to the output string T_{j+1}.
for j in range(M):
    # Join the characters in the current row to form a single string. This is the raw T_{j+1}.
    T_raw = "".join(output_matrix[j])
    # The problem requires that the output strings T_j does not end with the character '*'.
    # Use the rstrip() string method to remove any trailing '*' characters from the raw string.
    trimmed_T = T_raw.rstrip('*')
    # Print the resulting trimmed string T_{j+1}.
    print(trimmed_T)