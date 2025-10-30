import sys
data = sys.stdin.read().splitlines()
rows_with_piece = set()
cols_with_piece = set()
for i in range(8):
    for j in range(8):
        if data[i][j] == '#':
            rows_with_piece.add(i)
            cols_with_piece.add(j)
num_empty_rows = 8 - len(rows_with_piece)
num_empty_cols = 8 - len(cols_with_piece)
answer = num_empty_rows * num_empty_cols
print(answer)