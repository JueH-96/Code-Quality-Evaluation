import sys

def reconstruct_sequence(S):
    # Split the string by '|' and remove the first empty string
    parts = S.split('|')[1:]
    # Count the number of '-' in each part
    A = [part.count('-') for part in parts]
    # Print the result as space-separated values
    print(' '.join(map(str, A)))

# Read input from stdin
S = sys.stdin.read().strip()
# Reconstruct the sequence and print the output
reconstruct_sequence(S)