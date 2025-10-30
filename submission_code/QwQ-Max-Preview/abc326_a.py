# Read input
x, y = map(int, input().split())

# Calculate the difference
delta = y - x

# Check conditions for stairs usage
if delta > 0:
    if delta <= 2:
        print("Yes")
    else:
        print("No")
else:
    if abs(delta) <= 3:
        print("Yes")
    else:
        print("No")