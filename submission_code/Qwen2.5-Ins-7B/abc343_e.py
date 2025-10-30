# YOUR CODE HERE
V1, V2, V3 = map(int, input().split())

# Check if the given V1, V2, V3 can be satisfied
if V1 + 3 * V2 + 6 * V3 != 3 * 7**3:
    print("No")
else:
    a1, b1, c1 = 0, 0, 0
    a2, b2, c2 = 0, 7, 0
    a3, b3, c3 = 7, 0, 0
    print("Yes", a1, b1, c1, a2, b2, c2, a3, b3, c3)