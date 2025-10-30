import sys

# Read input from stdin
data = sys.stdin.read().split()
S = data[0]
T = data[1]

# Find positions where S and T differ
diff_pos = [i for i in range(len(S)) if S[i] != T[i]]
M = len(diff_pos)

# Output M
print(M)

# If M > 0, perform the transformations
if M > 0:
    rem_pos = list(diff_pos)  # List of remaining positions to change
    curr = S  # Current string, start with S
    
    for _ in range(M):
        # Create list of candidates with their new strings after hypothetical change
        candidates = [(k, curr[:k] + T[k] + curr[k+1:]) for k in rem_pos]
        # Find the candidate with the smallest new string
        min_cand = min(candidates, key=lambda x: x[1])
        best_k, best_str = min_cand
        # Output the new string after change
        print(best_str)
        # Update current string to the new string
        curr = best_str
        # Remove the changed position from the remaining positions
        rem_pos.remove(best_k)