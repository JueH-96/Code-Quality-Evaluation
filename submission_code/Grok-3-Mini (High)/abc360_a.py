# Read the input string from stdin
S = input()

# Find the indices of 'R' and 'M'
idx_R = S.index('R')
idx_M = S.index('M')

# Check if 'R' is to the left of 'M' and print the result
if idx_R < idx_M:
    print("Yes")
else:
    print("No")