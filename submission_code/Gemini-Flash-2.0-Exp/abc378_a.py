a = list(map(int, input().split()))

counts = {}
for x in a:
    if x not in counts:
        counts[x] = 0
    counts[x] += 1

ans = 0
for x in counts:
    ans += counts[x] // 2

print(ans)