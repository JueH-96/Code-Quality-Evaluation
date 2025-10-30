# Read the 8x8 grid
rows = [input().strip() for _ in range(8)]

# Calculate the number of empty rows
empty_rows = sum(1 for row in rows if all(c == '.' for c in row))

# Calculate the number of empty columns
empty_cols = 0
for j in range(8):
    column = [rows[i][j] for i in range(8)]
    if all(c == '.' for c in column):
        empty_cols += 1

# The result is the product of empty rows and empty columns
print(empty_rows * empty_cols)