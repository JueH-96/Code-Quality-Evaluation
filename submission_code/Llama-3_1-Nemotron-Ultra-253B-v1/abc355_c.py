n, t = map(int, input().split())
a_list = list(map(int, input().split()))

rows = [0] * (n + 1)
columns = [0] * (n + 1)
main_diag = 0
anti_diag = 0

for i in range(t):
    a = a_list[i]
    row = (a - 1) // n + 1
    col = a - n * (row - 1)
    
    # Check row
    rows[row] += 1
    if rows[row] == n:
        print(i + 1)
        exit()
    
    # Check column
    columns[col] += 1
    if columns[col] == n:
        print(i + 1)
        exit()
    
    # Check main diagonal
    if row == col:
        main_diag += 1
        if main_diag == n:
            print(i + 1)
            exit()
    
    # Check anti-diagonal
    if row + col == n + 1:
        anti_diag += 1
        if anti_diag == n:
            print(i + 1)
            exit()

print(-1)