# YOUR CODE HERE
S = input().strip()
rice_index = S.index('R')
miso_index = S.index('M')

if rice_index < miso_index:
    print("Yes")
else:
    print("No")