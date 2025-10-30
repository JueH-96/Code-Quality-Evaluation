def reconstruct_sequence(S):
    # Initialize an empty list to hold the sequence A
    A = []
    
    # Split the string S by the delimiter '|'
    parts = S.split('|')
    
    # Iterate through the parts, skipping the first and last empty strings
    for part in parts[1:-1]:
        # The length of each part corresponds to the value A_i
        A.append(len(part))
    
    # Print the result as a space-separated string
    print(' '.join(map(str, A)))

# Read input from standard input
import sys
input_string = sys.stdin.read().strip()
reconstruct_sequence(input_string)