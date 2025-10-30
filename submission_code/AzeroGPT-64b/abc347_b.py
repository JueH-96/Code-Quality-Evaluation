s = input()
n = len(s)
ans = 0
seen = set()

for i in range(n):
    for j in range(i + 1, n + 1):
        tmp = s[i:j]
        if tmp not in seen:
            ans += 1
            seen.add(tmp)

print(ans)