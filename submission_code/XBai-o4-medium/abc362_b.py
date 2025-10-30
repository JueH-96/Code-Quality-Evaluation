# Read input points
x1, y1 = map(int, input().split())
x2, y2 = map(int, input().split())
x3, y3 = map(int, input().split())

# Calculate squared distances between each pair of points
ab2 = (x1 - x2)**2 + (y1 - y2)**2
bc2 = (x2 - x3)**2 + (y2 - y3)**2
ac2 = (x1 - x3)**2 + (y1 - y3)**2

# Check for right triangle condition
if (ab2 + bc2 == ac2) or (ab2 + ac2 == bc2) or (bc2 + ac2 == ab2):
    print("Yes")
else:
    print("No")