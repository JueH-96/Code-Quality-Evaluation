# Read the input values
X, Y = map(int, input().split())

# Calculate the difference
diff = Y - X

if diff > 0:
    # Moving up
    if diff <= 2:
        print("Yes")
    else:
        print("No")
else:
    # Moving down, check if the absolute difference is <=3
    if (X - Y) <= 3:
        print("Yes")
    else:
        print("No")