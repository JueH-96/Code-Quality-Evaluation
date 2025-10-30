# YOUR CODE HERE
a, b = map(int, input().split())

# Check if the numbers are in the same row and adjacent horizontally
if (a % 3 == 1 and b == a + 1) or (a % 3 == 2 and b == a + 1):
    print("Yes")
else:
    print("No")