# YOUR CODE HERE
S = input().strip()

# Find positions of R and M
pos_R = S.index('R')
pos_M = S.index('M')

# Check if R is to the left of M (smaller index)
if pos_R < pos_M:
    print("Yes")
else:
    print("No")