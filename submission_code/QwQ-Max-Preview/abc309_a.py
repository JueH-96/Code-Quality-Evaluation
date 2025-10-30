# Read input
A, B = map(int, input().split())

# Check if B is next to A and A is not at the end of a row
if B == A + 1 and A % 3 != 0:
    print("Yes")
else:
    print("No")