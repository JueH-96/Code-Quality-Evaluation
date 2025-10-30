A, B = map(int, input().split())
board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i in range(3):
    for j in range(2):
        if (board[i][j] == A and board[i][j+1] == B) or (board[i][j] == B and board[i][j+1] == A):
            print("Yes")
            exit(0)
print("No")