# YOUR CODE HERE
import sys

# Read input from stdin
input_line = sys.stdin.read().strip()
S_AB, S_AC, S_BC = input_line.split()

# Determine the ages based on the given relationships
if S_AB == '<' and S_AC == '<' and S_BC == '<':
    middle_brother = 'B'
elif S_AB == '<' and S_AC == '<' and S_BC == '>':
    middle_brother = 'C'
elif S_AB == '<' and S_AC == '>' and S_BC == '<':
    middle_brother = 'A'
elif S_AB == '<' and S_AC == '>' and S_BC == '>':
    middle_brother = 'B'
elif S_AB == '>' and S_AC == '<' and S_BC == '<':
    middle_brother = 'A'
elif S_AB == '>' and S_AC == '<' and S_BC == '>':
    middle_brother = 'B'
elif S_AB == '>' and S_AC == '>' and S_BC == '<':
    middle_brother = 'C'
elif S_AB == '>' and S_AC == '>' and S_BC == '>':
    middle_brother = 'A'

# Print the middle brother
print(middle_brother)