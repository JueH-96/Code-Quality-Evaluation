# YOUR CODE HERE
a, b = map(int, input().split())

board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

pos_a = None
pos_b = None

for i in range(3):
    for j in range(3):
        if board[i][j] == a:
            pos_a = (i, j)
        if board[i][j] == b:
            pos_b = (i, j)

if pos_a is not None and pos_b is not None:
    if pos_a[0] == pos_b[0] and abs(pos_a[1] - pos_b[1]) == 1:
        print("Yes")
    else:
        print("No")