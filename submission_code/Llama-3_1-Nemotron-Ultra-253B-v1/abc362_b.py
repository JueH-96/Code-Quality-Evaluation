# Read the input points
xA, yA = map(int, input().split())
xB, yB = map(int, input().split())
xC, yC = map(int, input().split())

# Calculate squared distances
dAB = (xB - xA)**2 + (yB - yA)**2
dBC = (xC - xB)**2 + (yC - yB)**2
dCA = (xA - xC)**2 + (yA - yC)**2

# Check if any combination satisfies the Pythagorean theorem
if (dAB + dBC == dCA) or (dAB + dCA == dBC) or (dBC + dCA == dAB):
    print("Yes")
else:
    print("No")