# NUMBER OF SLIMES
n = int(input())
slimes = []
for _ in range(n):
    s, c = map(int, input().split())
    slimes.append((s, c))

slimes.sort()

min_slimes = 0
for s, c in slimes:
    if c % 2 == 1:
        min_slimes += 1
    c //= 2
    if c > 0:
        min_slimes += c

print(min_slimes)