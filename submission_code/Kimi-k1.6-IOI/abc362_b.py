# Read the input points
xA, yA = map(int, input().split())
xB, yB = map(int, input().split())
xC, yC = map(int, input().split())

# Calculate the squared lengths of the sides
AB_sq = (xB - xA)**2 + (yB - yA)**2
BC_sq = (xC - xB)**2 + (yC - yB)**2
AC_sq = (xC - xA)**2 + (yC - yA)**2

# Check if any of the three conditions for a right triangle are met
if (AB_sq + BC_sq == AC_sq) or (AB_sq + AC_sq == BC_sq) or (BC_sq + AC_sq == AB_sq):
    print("Yes")
else:
    print("No")