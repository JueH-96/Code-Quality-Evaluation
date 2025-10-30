# YOUR CODE HERE
X, Y = map(int, input().split())

if Y > X:  # Going up
    if Y - X <= 2:
        print("Yes")
    else:
        print("No")
else:  # Going down (Y < X, since X ≠ Y is guaranteed)
    if X - Y <= 3:
        print("Yes")
    else:
        print("No")