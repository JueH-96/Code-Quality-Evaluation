# YOUR CODE HERE
X, Y = map(int, input().split())

if Y > X:  # moving up
    if Y - X <= 2:
        print("Yes")
    else:
        print("No")
else:  # moving down (Y < X)
    if X - Y <= 3:
        print("Yes")
    else:
        print("No")