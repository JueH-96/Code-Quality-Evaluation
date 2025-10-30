# YOUR CODE HERE
import sys

def reconstruct_sequence(S):
    # Split the string by '|' and filter out empty strings
    parts = S.split('|')
    parts = [part for part in parts if part]
    
    # The length of each part corresponds to the elements in A
    A = [len(part) for part in parts]
    
    return A

# Read input from stdin
S = sys.stdin.read().strip()

# Get the sequence A
A = reconstruct_sequence(S)

# Print the result
print(' '.join(map(str, A)))