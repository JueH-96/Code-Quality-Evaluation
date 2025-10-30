# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

indices = {}
for i in range(3 * n):
    if a[i] not in indices:
        indices[a[i]] = []
    indices[a[i]].append(i)

f = {}
for i in range(1, n + 1):
    f[i] = indices[i][1]

sorted_keys = sorted(f.keys(), key=lambda x: f[x])

print(*sorted_keys)