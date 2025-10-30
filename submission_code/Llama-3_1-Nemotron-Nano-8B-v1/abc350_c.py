n = int(input())
a = list(map(int, input().split()))
pos = [0] * (n + 1)
for i in range(n):
    pos[a[i]] = i
swaps = []
for i in range(n):
    current = a[i]
    target = i + 1
    if current == target:
        continue
    j = pos[target]
    swaps.append((i + 1, j + 1))
    a[i], a[j] = a[j], a[i]
    pos[current] = j
    pos[target] = i
print(len(swaps))
for swap in swaps:
    print(swap[0], swap[1])