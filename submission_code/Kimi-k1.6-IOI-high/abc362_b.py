# Read input coordinates
xa, ya = map(int, input().split())
xb, yb = map(int, input().split())
xc, yc = map(int, input().split())

# Calculate squared distances between each pair of points
ab_sq = (xb - xa)**2 + (yb - ya)**2
bc_sq = (xc - xb)**2 + (yc - yb)**2
ca_sq = (xa - xc)**2 + (ya - yc)**2

# Sort the squared distances
squared_distances = [ab_sq, bc_sq, ca_sq]
squared_distances.sort()

# Check if the sum of the two smaller squared distances equals the largest one
if squared_distances[0] + squared_distances[1] == squared_distances[2]:
    print("Yes")
else:
    print("No")