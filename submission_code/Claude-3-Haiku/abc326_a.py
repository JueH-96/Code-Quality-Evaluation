# Read the input from stdin
x, y = map(int, input().split())

# Check if Takahashi uses the stairs
if abs(x - y) <= 3:
    print("Yes")
else:
    print("No")