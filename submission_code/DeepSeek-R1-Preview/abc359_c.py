# Read input
Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

# Calculate transformed coordinates for start and end points
a1 = (Sx + Sy) // 2
b1 = (Sx - Sy) // 2
a2 = (Tx + Ty) // 2
b2 = (Tx - Ty) // 2

# Compute Manhattan distance
toll = abs(a1 - a2) + abs(b1 - b2)

# Output the result
print(toll)