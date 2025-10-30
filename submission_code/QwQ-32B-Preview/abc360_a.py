# Read the input string
S = input().strip()

# Find the indices of 'R' and 'M'
index_R = S.index('R')
index_M = S.index('M')

# Compare the indices and print the result
if index_R < index_M:
    print("Yes")
else:
    print("No")