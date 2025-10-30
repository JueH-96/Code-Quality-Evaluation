n, t = map(int, input().split())
a_list = list(map(int, input().split()))

row_counts = [0] * (n + 1)
col_counts = [0] * (n + 1)
main_diag = 0
anti_diag = 0

for step in range(t):
    a = a_list[step]
    i = (a - 1) // n + 1
    j = (a - 1) % n + 1
    
    # Update row count and check
    row_counts[i] += 1
    if row_counts[i] == n:
        print(step + 1)
        exit()
    
    # Update column count and check
    col_counts[j] += 1
    if col_counts[j] == n:
        print(step + 1)
        exit()
    
    # Update main diagonal if applicable and check
    if i == j:
        main_diag += 1
        if main_diag == n:
            print(step + 1)
            exit()
    
    # Update anti-diagonal if applicable and check
    if i + j == n + 1:
        anti_diag += 1
        if anti_diag == n:
            print(step + 1)
            exit()

print(-1)