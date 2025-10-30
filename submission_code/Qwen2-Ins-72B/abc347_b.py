s = input()
n = len(s)
ans = 0
seen = set()
for i in range(n):
    for j in range(i+1, n+1):
        t = s[i:j]
        if t not in seen:
            ans += 1
            seen.add(t)
print(ans)