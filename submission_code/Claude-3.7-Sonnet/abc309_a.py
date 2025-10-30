A, B = map(int, input().split())

# Check if A and B are in the same row
same_row = ((A - 1) // 3 == (B - 1) // 3)

# Check if they are consecutive numbers
differ_by_1 = (B - A == 1)

if same_row and differ_by_1:
    print("Yes")
else:
    print("No")