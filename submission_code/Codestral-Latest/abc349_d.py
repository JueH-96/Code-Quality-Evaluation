# YOUR CODE HERE
import sys
import math

def find_good_sequences(L, R):
    sequences = []
    while L < R:
        # Find the largest power of 2 that is less than or equal to R - L
        power = 2 ** int(math.log2(R - L))
        # Find the largest multiple of power that is less than or equal to L
        start = (L // power) * power
        # Find the smallest multiple of power that is greater than L
        end = start + power
        sequences.append((start, end))
        L = end
    return sequences

# Read input
L, R = map(int, sys.stdin.read().split())

# Find good sequences
sequences = find_good_sequences(L, R)

# Print the number of sequences
print(len(sequences))
# Print each sequence
for seq in sequences:
    print(seq[0], seq[1])