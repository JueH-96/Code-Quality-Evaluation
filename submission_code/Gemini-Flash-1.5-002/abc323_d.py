# YOUR CODE HERE
n = int(input())
slimes = {}
for i in range(n):
    s, c = map(int, input().split())
    slimes[s] = c

ans = 0
for s, c in slimes.items():
    if c % 2 == 0:
        ans += c // 2
    else:
        ans += c // 2 + 1

print(ans)