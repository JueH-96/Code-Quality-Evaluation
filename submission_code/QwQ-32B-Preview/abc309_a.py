def get_position(n):
    row = (n - 1) // 3
    col = (n - 1) % 3
    return row, col

def are_adjacent_horizontal(A, B):
    row_A, col_A = get_position(A)
    row_B, col_B = get_position(B)
    if row_A == row_B and abs(col_A - col_B) == 1:
        return "Yes"
    else:
        return "No"

# Read input
A, B = map(int, input().split())

# Get and print the result
print(are_adjacent_horizontal(A, B))