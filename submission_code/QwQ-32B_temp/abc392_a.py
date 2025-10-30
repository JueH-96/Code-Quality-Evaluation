# Read the input values
a, b, c = map(int, input().split())

# Check all permutations to see if any satisfy B1 * B2 = B3
if (a * b == c) or (a * c == b) or (b * a == c) or (b * c == a) or (c * a == b) or (c * b == a):
    print("Yes")
else:
    print("No")