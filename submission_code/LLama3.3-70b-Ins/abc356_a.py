import sys

def reverse_sequence(N, L, R):
    # Create the initial sequence
    sequence = list(range(1, N + 1))
    
    # Reverse the L-th through R-th elements
    sequence[L-1:R] = reversed(sequence[L-1:R])
    
    # Print the sequence after the operation
    print(' '.join(map(str, sequence)))

# Read the inputs from stdin
N, L, R = map(int, sys.stdin.readline().split())

# Solve the problem and write the answer to stdout
reverse_sequence(N, L, R)