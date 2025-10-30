A, B = map(int, input().split())
a_row = (A - 1) // 3
a_col = (A - 1) % 3
b_row = (B - 1) // 3
b_col = (B - 1) % 3
if a_row == b_row and abs(a_col - b_col) == 1:
    print("Yes")
else:
    print("No")