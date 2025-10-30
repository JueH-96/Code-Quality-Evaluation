n, k = map(int, input().split())
s = list(input())
count = 0
ans = 0
for i in range(n):
    if s[i] == 'O':
        count += 1
    else:
        count = 0
    if count == k:
        ans += 1
        for j in range(i - k + 1, i + 1):
            s[j] = 'X'
        count = 0
print(ans)