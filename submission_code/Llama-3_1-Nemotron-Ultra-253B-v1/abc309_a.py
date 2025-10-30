A, B = map(int, input().split())

def get_row_col(n):
    row = (n - 1) // 3
    col = (n - 1) % 3
    return row, col

ra, ca = get_row_col(A)
rb, cb = get_row_col(B)

if ra == rb and abs(ca - cb) == 1:
    print("Yes")
else:
    print("No")