X, Y = map(int, input().split())

if Y > X:
    diff = Y - X
    if diff <= 2:
        print("Yes")
    else:
        print("No")
else:
    diff = X - Y
    if diff <= 3:
        print("Yes")
    else:
        print("No")