A, B = map(int, input().split())

def get_position(x):
    row = (x - 1) // 3
    col = (x - 1) % 3
    return (row, col)

row_a, col_a = get_position(A)
row_b, col_b = get_position(B)

if row_a == row_b and abs(col_a - col_b) == 1:
    print("Yes")
else:
    print("No")