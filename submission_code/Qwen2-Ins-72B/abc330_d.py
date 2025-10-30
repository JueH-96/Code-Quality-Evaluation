import sys
n = int(input())
s = [list(sys.stdin.readline().strip()) for _ in range(n)]
ans = 0
for i in range(n):
    for j in range(n):
        if s[i][j] == 'o':
            ans += (n - 1 - s[i].count('x')) * (s[j][0:i].count('o') + s[j][i + 1:n].count('o'))
print(ans)