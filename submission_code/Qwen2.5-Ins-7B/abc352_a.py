# YOUR CODE HERE
N, X, Y, Z = map(int, input().split())
if X < Y:
    print('Yes' if X <= Z <= Y else 'No')
else:
    print('Yes' if Y <= Z <= X else 'No')