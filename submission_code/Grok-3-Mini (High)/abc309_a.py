# Read input from stdin
A, B = map(int, input().split())

# Check if A and B are in the same row and adjacent horizontally
if (A - 1) // 3 == (B - 1) // 3 and abs(A - B) == 1:
    print("Yes")
else:
    print("No")