import sys

# Read input from stdin
data = sys.stdin.read().split()

# Extract coordinates
xA = int(data[0])
yA = int(data[1])
xB = int(data[2])
yB = int(data[3])
xC = int(data[4])
yC = int(data[5])

# Compute squared distances between points
dist_AB_sq = (xB - xA)**2 + (yB - yA)**2
dist_BC_sq = (xC - xB)**2 + (yC - yB)**2
dist_AC_sq = (xC - xA)**2 + (yC - yA)**2

# Store squared distances in a list and sort it
len_sq = [dist_AB_sq, dist_BC_sq, dist_AC_sq]
sorted_len = sorted(len_sq)

# Check Pythagorean theorem: sum of squares of two smaller sides equals square of largest side
if sorted_len[0] + sorted_len[1] == sorted_len[2]:
    print("Yes")
else:
    print("No")