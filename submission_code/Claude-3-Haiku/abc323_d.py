from collections import defaultdict

N = int(input())
sizes = defaultdict(int)
for _ in range(N):
    s, c = map(int, input().split())
    sizes[s] += c

slimes = []
for s, c in sorted(sizes.items()):
    slimes.extend([s] * c)

ans = 0
while len(slimes) > 1:
    slimes.sort()
    if slimes[-1] == slimes[-2]:
        slimes.pop()
        slimes.pop()
        slimes.append(slimes[-1] * 2)
    else:
        ans += len(slimes) - 1
        break

print(ans + len(slimes))