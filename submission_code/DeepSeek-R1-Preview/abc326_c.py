n, m = map(int, input().split())
a = list(map(int, input().split()))
a.sort()

max_count = 0
j = 0

for i in range(n):
    while j < n and a[j] - a[i] < m:
        j += 1
    current = j - i
    if current > max_count:
        max_count = current

print(max_count)