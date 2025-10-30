# YOUR CODE HERE
N, X, Y, Z = map(int, input().split())

if min(X, Y) <= Z <= max(X, Y):
    print("Yes")
else:
    print("No")