# Read the three points from input
A = tuple(map(int, input().split()))
B = tuple(map(int, input().split()))
C = tuple(map(int, input().split()))

# Calculate the squared distances between each pair of points
dAB_sq = (B[0] - A[0])**2 + (B[1] - A[1])**2
dBC_sq = (C[0] - B[0])**2 + (C[1] - B[1])**2
dAC_sq = (C[0] - A[0])**2 + (C[1] - A[1])**2

# Check if any of the three conditions for a right triangle hold
if (dAB_sq + dAC_sq == dBC_sq) or (dAB_sq + dBC_sq == dAC_sq) or (dAC_sq + dBC_sq == dAB_sq):
    print("Yes")
else:
    print("No")