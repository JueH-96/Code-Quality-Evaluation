# Read input
Sx, Sy = map(int, input().split())
Tx, Ty = map(int, input().split())

# Compute transformed coordinates for start and end
a1 = (Sx + Sy) // 2
b1 = (Sx - Sy) // 2
a2 = (Tx + Ty) // 2
b2 = (Tx - Ty) // 2

# Calculate Manhattan distance
result = abs(a1 - a2) + abs(b1 - b2)

# Output the result
print(result)