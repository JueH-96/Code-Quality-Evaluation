# YOUR CODE HERE

# Read the input
X, Y = map(int, input().split())

# Calculate the difference between the floors
diff = abs(X - Y)

# Check if the difference is less than or equal to 2
if diff <= 2:
    print("Yes")
else:
    print("No")