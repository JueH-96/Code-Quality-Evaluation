n, d = map(int, input().split())
s = [input() for _ in range(n)]
ans = 0
for i in range(d):
    cnt = 0
    for j in range(i, d):
        for k in range(n):
            if s[k][j] == 'x':
                break
        else:
            cnt += 1
        if k == n-1:
            ans = max(ans, cnt)
print(ans)