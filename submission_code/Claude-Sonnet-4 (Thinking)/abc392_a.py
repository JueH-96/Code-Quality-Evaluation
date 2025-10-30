# YOUR CODE HERE
# Read input
a, b, c = map(int, input().split())

# Check all possible products
if a * b == c or a * c == b or b * c == a:
    print("Yes")
else:
    print("No")