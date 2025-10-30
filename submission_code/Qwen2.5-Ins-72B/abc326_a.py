# YOUR CODE HERE
X, Y = map(int, input().split())

if (Y - X) <= 2 or (X - Y) <= 3:
    print("Yes")
else:
    print("No")