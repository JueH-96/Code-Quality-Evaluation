n, t = map(int, input().split())
a_list = list(map(int, input().split()))

row = [0] * (n + 1)
col = [0] * (n + 1)
diag1 = 0
diag2 = 0
ans = -1

for turn in range(t):
    a = a_list[turn]
    i = (a - 1) // n + 1
    j = (a - 1) % n + 1
    
    row[i] += 1
    if row[i] == n:
        ans = turn + 1
        break
    
    col[j] += 1
    if col[j] == n:
        ans = turn + 1
        break
    
    if i == j:
        diag1 += 1
        if diag1 == n:
            ans = turn + 1
            break
    
    if i + j == n + 1:
        diag2 += 1
        if diag2 == n:
            ans = turn + 1
            break

print(ans)