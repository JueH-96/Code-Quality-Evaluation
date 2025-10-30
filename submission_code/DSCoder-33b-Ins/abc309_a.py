# YOUR CODE HERE

# Read the inputs
A, B = map(int, input().split())

# Check if the squares are adjacent horizontally
if abs(A - B) == 1:
    print("Yes")
else:
    print("No")