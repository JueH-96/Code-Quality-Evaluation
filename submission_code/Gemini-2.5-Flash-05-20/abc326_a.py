# Read X and Y from standard input
X, Y = map(int, input().split())

# Calculate the difference in floors
diff = Y - X

# Determine if Takahashi uses the stairs or elevator
if diff > 0:  # Moving up
    # Stairs if moving up two floors or less
    if diff <= 2:
        print("Yes")
    else:
        print("No")
elif diff < 0:  # Moving down
    # Stairs if moving down three floors or less
    # The absolute difference is -diff (since diff is negative)
    if -diff <= 3:
        print("Yes")
    else:
        print("No")

# The problem states X != Y, so diff will never be 0.