# YOUR CODE HERE

# Read the input string
S = input()

# Check if S is an Extended ABC string
if S.count('A') == len(S) and S.count('B') == len(S) and S.count('C') == len(S):
    print('Yes')
else:
    print('No')