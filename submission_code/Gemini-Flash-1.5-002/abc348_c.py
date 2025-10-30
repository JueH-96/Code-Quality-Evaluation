# YOUR CODE HERE
n = int(input())
colors = {}
for i in range(n):
    a, c = map(int, input().split())
    if c not in colors:
        colors[c] = []
    colors[c].append(a)

ans = 0
for c in colors:
    ans = max(ans, min(colors[c]))

print(ans)