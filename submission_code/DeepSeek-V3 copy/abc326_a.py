# Read input values for X and Y
X, Y = map(int, input().split())

# Calculate the difference between Y and X
difference = Y - X

# Determine if Takahashi uses the stairs
if difference > 0 and difference <= 2:
    print("Yes")
elif difference < 0 and difference >= -3:
    print("Yes")
else:
    print("No")