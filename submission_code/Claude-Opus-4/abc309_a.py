# YOUR CODE HERE
A, B = map(int, input().split())

# Check if they are in the same row and consecutive
if (A - 1) // 3 == (B - 1) // 3 and B == A + 1:
    print("Yes")
else:
    print("No")