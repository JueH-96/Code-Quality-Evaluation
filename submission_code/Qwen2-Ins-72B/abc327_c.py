A = [list(map(int, input().split())) for _ in range(9)]

def check_row():
    for row in A:
        if len(set(row)) != 9:
            return False
    return True

def check_column():
    for j in range(9):
        column = [A[i][j] for i in range(9)]
        if len(set(column)) != 9:
            return False
    return True

def check_square():
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = [A[i+k][j+l] for k in range(3) for l in range(3)]
            if len(set(square)) != 9:
                return False
    return True

if check_row() and check_column() and check_square():
    print("Yes")
else:
    print("No")