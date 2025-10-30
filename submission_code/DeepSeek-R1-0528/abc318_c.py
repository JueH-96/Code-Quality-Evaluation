n, d, p = map(int, input().split())
fares = list(map(int, input().split()))
total = sum(fares)
fares.sort(reverse=True)

prefix = [0]
for x in fares:
    prefix.append(prefix[-1] + x)

k_max = (n + d - 1) // d
ans = total

for k in range(1, k_max + 1):
    cover = min(k * d, n)
    cost = k * p + (total - prefix[cover])
    if cost < ans:
        ans = cost

print(ans)