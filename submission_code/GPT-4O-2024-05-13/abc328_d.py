# YOUR CODE HERE
def remove_abc_substrings(S):
    while 'ABC' in S:
        S = S.replace('ABC', '', 1)
    return S

# Read input from stdin
import sys
input = sys.stdin.read().strip()

# Process the input and print the result
result = remove_abc_substrings(input)
print(result)