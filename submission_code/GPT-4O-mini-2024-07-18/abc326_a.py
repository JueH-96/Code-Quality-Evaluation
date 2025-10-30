# YOUR CODE HERE
X, Y = map(int, input().split())

if abs(X - Y) <= 2 or abs(X - Y) == 3:
    print("Yes")
else:
    print("No")