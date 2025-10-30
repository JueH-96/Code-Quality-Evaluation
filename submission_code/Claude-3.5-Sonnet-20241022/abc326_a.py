# Read input
X, Y = map(int, input().split())

# Calculate the difference in floors
diff = Y - X

# Check conditions for using stairs
if diff > 0:  # Going up
    if diff <= 2:
        print("Yes")
    else:
        print("No")
else:  # Going down
    if abs(diff) <= 3:
        print("Yes")
    else:
        print("No")