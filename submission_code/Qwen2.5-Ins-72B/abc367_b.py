# YOUR CODE HERE
X = input().strip()
if X.endswith('.000'):
    print(int(float(X)))
else:
    print(float(X))