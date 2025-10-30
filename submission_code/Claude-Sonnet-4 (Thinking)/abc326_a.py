# YOUR CODE HERE
X, Y = map(int, input().split())

if Y > X:  # going up
    floors_up = Y - X
    if floors_up <= 2:
        print("Yes")
    else:
        print("No")
else:  # going down (Y < X)
    floors_down = X - Y
    if floors_down <= 3:
        print("Yes")
    else:
        print("No")