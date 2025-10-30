# YOUR CODE HERE
X = float(input())

if X.is_integer():
    print(int(X))
else:
    print(format(X, '.3f').rstrip("0").rstrip("."))