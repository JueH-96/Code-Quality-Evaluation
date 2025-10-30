# YOUR CODE HERE

# Read the input
S = input()

# Check if the rice is to the left of the miso soup
if S.index('R') < S.index('M'):
    print('Yes')
else:
    print('No')