# Read the input from stdin
a, b = map(int, input().split())

# Check if the two numbers are adjacent horizontally
if abs(a - b) == 1:
    print("Yes")
else:
    print("No")