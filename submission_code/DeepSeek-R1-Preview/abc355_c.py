n, t = map(int, input().split())
a = list(map(int, input().split()))

rows = [0] * n
cols = [0] * n
diag1 = 0
diag2 = 0

for turn in range(t):
    num = a[turn]
    i = (num - 1) // n
    j = (num - 1) % n
    
    # Update row
    rows[i] += 1
    if rows[i] == n:
        print(turn + 1)
        exit()
    
    # Update column
    cols[j] += 1
    if cols[j] == n:
        print(turn + 1)
        exit()
    
    # Check main diagonal
    if i == j:
        diag1 += 1
        if diag1 == n:
            print(turn + 1)
            exit()
    
    # Check anti-diagonal
    if i + j == n - 1:
        diag2 += 1
        if diag2 == n:
            print(turn + 1)
            exit()

print(-1)