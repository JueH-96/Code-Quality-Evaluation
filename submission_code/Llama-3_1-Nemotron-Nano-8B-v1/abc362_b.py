# Read the coordinates of points A, B, and C
a = list(map(int, input().split()))
b = list(map(int, input().split()))
c = list(map(int, input().split()))

# Calculate squared distances between each pair of points
dAB = (b[0] - a[0])**2 + (b[1] - a[1])**2
dBC = (c[0] - b[0])**2 + (c[1] - b[1])**2
dCA = (a[0] - c[0])**2 + (a[1] - c[1])**2

# Check if any combination satisfies the Pythagorean theorem
if (dAB + dBC == dCA) or (dAB + dCA == dBC) or (dBC + dCA == dAB):
    print("Yes")
else:
    print("No")