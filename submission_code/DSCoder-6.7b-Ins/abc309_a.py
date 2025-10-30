# YOUR CODE HERE

# Read the inputs
A, B = map(int, input().split())

# Check if the numbers are adjacent horizontally
if (B - A) == 1:
    print("Yes")
else:
    print("No")