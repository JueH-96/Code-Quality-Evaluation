# Read the input values
X, Y = map(int, input().split())

# Check if moving up or down and determine the mode of transport
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