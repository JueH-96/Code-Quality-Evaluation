# YOUR CODE HERE
a, b = map(int, input().split())

row_a = (a - 1) // 3
col_a = (a - 1) % 3
row_b = (b - 1) // 3
col_b = (b - 1) % 3

if row_a == row_b and abs(col_a - col_b) == 1:
    print("Yes")
else:
    print("No")