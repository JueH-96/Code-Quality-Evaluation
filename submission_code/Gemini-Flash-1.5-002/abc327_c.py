def solve():
    A = []
    for _ in range(9):
        A.append(list(map(int, input().split())))

    def check_rows():
        for row in A:
            if len(set(row)) != 9 or max(row) > 9 or min(row) < 1:
                return False
        return True

    def check_cols():
        for col in range(9):
            column = [A[row][col] for row in range(9)]
            if len(set(column)) != 9 or max(column) > 9 or min(column) < 1:
                return False
        return True

    def check_subgrids():
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = []
                for row in range(i, i + 3):
                    for col in range(j, j + 3):
                        subgrid.append(A[row][col])
                if len(set(subgrid)) != 9 or max(subgrid) > 9 or min(subgrid) < 1:
                    return False
        return True

    if check_rows() and check_cols() and check_subgrids():
        print("Yes")
    else:
        print("No")

solve()