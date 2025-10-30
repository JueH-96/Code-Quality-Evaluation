# YOUR CODE HERE

import sys

def find_sequence(n, k):
    # Initialize the sequence
    sequence = []
    for i in range(1, n+1):
        sequence += [i]*k

    # Find the middle sequence
    middle = (len(sequence)+1)//2
    sequence = sequence[:middle]

    # Sort the sequence
    sequence.sort()

    return sequence

# Read the inputs
n, k = map(int, sys.stdin.readline().split())

# Find the sequence
sequence = find_sequence(n, k)

# Print the sequence
print(' '.join(map(str, sequence)))