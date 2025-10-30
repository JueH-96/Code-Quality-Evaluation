# YOUR CODE HERE
n = int(input())
events = []
for _ in range(n):
    w, x = map(int, input().split())
    events.append((max(9 - x, 0), w))
    events.append((min(18 - x, 24) + 1, -w))
events.sort()

ans, cur = 0, 0
for t, w in events:
    cur += w
    ans = max(ans, cur)

print(ans)