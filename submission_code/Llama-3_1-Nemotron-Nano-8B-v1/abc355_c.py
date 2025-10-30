n, t = map(int, input().split())
a_list = list(map(int, input().split()))

rows = [0] * (n + 1)
cols = [0] * (n + 1)
main_diag = 0
anti_diag = 0

for turn in range(1, t + 1):
    a = a_list[turn - 1]
    i = (a - 1) // n + 1
    j = (a - 1) % n + 1
    
    # Check row
    rows[i] += 1
    if rows[i] == n:
        print(turn)
        exit()
    
    # Check column
    cols[j] += 1
    if cols[j] == n:
        print(turn)
        exit()
    
    # Check main diagonal
    if i == j:
        main_diag += 1
        if main_diag == n:
            print(turn)
            exit()
    
    # Check anti-diagonal
    if i + j == n + 1:
        anti_diag += 1
        if anti_diag == n:
            print(turn)
            exit()

print(-1)