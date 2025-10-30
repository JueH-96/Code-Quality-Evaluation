# YOUR CODE HERE
import sys

def find_first_difference(S, T):
    len_s = len(S)
    len_t = len(T)
    
    # Find the first differing character
    for i in range(min(len_s, len_t)):
        if S[i] != T[i]:
            return i + 1
    
    # If one string is a prefix of the other, the first differing position is the next character
    if len_s != len_t:
        return min(len_s, len_t) + 1
    
    # If no differences found and lengths are the same, return 0
    return 0

# Read input from stdin
input = sys.stdin.read().strip().split()
S = input[0]
T = input[1]

# Find and print the first differing position
print(find_first_difference(S, T))