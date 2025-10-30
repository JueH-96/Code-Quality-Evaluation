# Read the three points
xA, yA = map(int, input().split())
xB, yB = map(int, input().split())
xC, yC = map(int, input().split())

# Calculate squared distances between each pair of points
ab_sq = (xB - xA)**2 + (yB - yA)**2
ac_sq = (xC - xA)**2 + (yC - yA)**2
bc_sq = (xC - xB)**2 + (yC - yB)**2

# Check the three possible conditions for a right triangle
if (ab_sq + ac_sq == bc_sq) or (ab_sq + bc_sq == ac_sq) or (ac_sq + bc_sq == ab_sq):
    print("Yes")
else:
    print("No")