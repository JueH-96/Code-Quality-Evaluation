# YOUR CODE HERE
a = list(map(int, input().split()))
counts = {}
for x in a:
    counts[x] = counts.get(x, 0) + 1

ans = 0
for x in counts:
    ans += counts[x] // 2

print(ans)