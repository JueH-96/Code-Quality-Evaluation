# Read input values
X, Y = map(int, input().split())

# Check conditions for using stairs
if Y > X:
    if Y - X <= 2:
        print("Yes")
    else:
        print("No")
else:
    if X - Y <= 3:
        print("Yes")
    else:
        print("No")