n = int(input())
a = list(map(int, input().split()))
# Convert to 1-based indexing
a = [0] + a

prev = [0] * (n + 2)
next = [n + 1] * (n + 2)

last_occurrence = {}
for i in range(1, n + 1):
    x = a[i]
    prev[i] = last_occurrence.get(x, 0)
    last_occurrence[x] = i

last_occurrence = {}
for i in range(n, 0, -1):
    x = a[i]
    next_val = last_occurrence.get(x, n + 1)
    next[i] = next_val
    last_occurrence[x] = i

total = 0
for i in range(1, n + 1):
    total += i * (next[i] - i)

print(total)