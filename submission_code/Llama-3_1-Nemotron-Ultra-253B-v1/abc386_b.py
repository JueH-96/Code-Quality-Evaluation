s = input().strip()
ans = 0
i = 0
n = len(s)
while i < n:
    if s[i] == '0' and i + 1 < n and s[i+1] == '0':
        ans += 1
        i += 2
    else:
        ans += 1
        i += 1
print(ans)