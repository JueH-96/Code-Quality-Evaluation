# YOUR CODE HERE
a, b = map(int, input().split())
board = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
for i in range(9):
    if board[i][0] == a and board[i][1] == b:
        if abs(board[i][1] - board[i-1][1]) == 1 and board[i][0] == board[i-1][0]:
            print('Yes')
            break
else:
    print('No')