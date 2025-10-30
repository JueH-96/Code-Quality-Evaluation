n = int(input())
a = list(map(int, input().split()))
pos = [0] * (n + 1)
for i in range(n):
    pos[a[i]] = i
swaps = []
for i in range(n):
    x = i + 1
    if a[i] == x:
        continue
    j = pos[x]
    a[i], a[j] = a[j], a[i]
    pos[a[i]] = i
    pos[a[j]] = j
    swaps.append((i + 1, j + 1))
print(len(swaps))
for swap in swaps:
    print(swap[0], swap[1])