# YOUR CODE HERE

# Read the inputs
W, B = map(int, input().split())

# Create the string S
S = "wb" * (W + B)

# Check if there is a substring of S that consists of W occurrences of w and B occurrences of b
if S.count('w'*W + 'b'*B) > 0:
    print('Yes')
else:
    print('No')