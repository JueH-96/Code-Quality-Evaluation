# YOUR CODE HERE
S = input().strip()

# Find the positions of 'R' and 'M'
pos_R = S.index('R')
pos_M = S.index('M')

# Check if 'R' is to the left of 'M'
if pos_R < pos_M:
    print("Yes")
else:
    print("No")