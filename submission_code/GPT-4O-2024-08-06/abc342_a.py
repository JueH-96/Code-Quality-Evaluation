# YOUR CODE HERE
import sys

def find_different_character_index(S):
    # We know that all characters but one are the same
    # We can check the first three characters to determine the common character
    if S[0] == S[1]:
        common_char = S[0]
    elif S[0] == S[2]:
        common_char = S[0]
    else:
        common_char = S[1]
    
    # Find the index of the character that is not the common character
    for i in range(len(S)):
        if S[i] != common_char:
            return i + 1  # Return 1-based index

# Read input
S = sys.stdin.read().strip()

# Find and print the index of the different character
print(find_different_character_index(S))