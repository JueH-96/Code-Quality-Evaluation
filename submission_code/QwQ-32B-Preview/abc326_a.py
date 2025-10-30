X, Y = map(int, input().split())

if Y > X and (Y - X) <= 2:
    print("Yes")
elif Y < X and (X - Y) <= 3:
    print("Yes")
else:
    print("No")