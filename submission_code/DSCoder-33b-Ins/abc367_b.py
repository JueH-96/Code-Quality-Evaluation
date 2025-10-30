# YOUR CODE HERE

X = float(input())

if X.is_integer():
    print(int(X))
else:
    print('{:.3f}'.format(X).rstrip('0').rstrip('.'))