A, B = map(int, input().split())

grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

number_positions = {}
for row_idx in range(3):
    for col_idx in range(3):
        number_positions[grid[row_idx][col_idx]] = (row_idx, col_idx)

a_row, a_col = number_positions[A]
b_row, b_col = number_positions[B]

if a_row == b_row and abs(a_col - b_col) == 1:
    print("Yes")
else:
    print("No")