# YOUR CODE HERE
def find_different_char_index(S):
    # Find the character that is different
    if S[0] != S[1] and S[0] != S[2]:
        return 1
    for i in range(1, len(S)):
        if S[i] != S[0]:
            return i + 1

# Read input
import sys
input = sys.stdin.read().strip()

# Find and print the result
print(find_different_char_index(input))