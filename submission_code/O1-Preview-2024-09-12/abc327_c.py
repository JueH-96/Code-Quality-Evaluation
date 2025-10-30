# YOUR CODE HERE
A = []
for _ in range(9):
    A.append(list(map(int, input().split())))
for row in A:
    if set(row) != set(range(1, 10)):
        print('No')
        exit()
for j in range(9):
    column = [A[i][j] for i in range(9)]
    if set(column) != set(range(1, 10)):
        print('No')
        exit()
for I in [0, 3, 6]:
    for J in [0, 3, 6]:
        block = []
        for i in range(I, I+3):
            for j in range(J, J+3):
                block.append(A[i][j])
        if set(block) != set(range(1, 10)):
            print('No')
            exit()
print('Yes')