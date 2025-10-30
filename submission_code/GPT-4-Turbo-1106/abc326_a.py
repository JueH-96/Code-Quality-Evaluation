X, Y = map(int, input().split())

# Calculate the difference in floors
difference = abs(X - Y)

# Determine if Takahashi uses the stairs or the elevator
if difference <= 2 or (X > Y and difference <= 3):
    print("Yes")
else:
    print("No")