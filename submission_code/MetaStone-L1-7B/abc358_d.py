n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a.sort()
b.sort()

j = 0
total = 0

for bi in b:
    while j < n and a[j] < bi:
        j += 1
    if j >= n:
        print(-1)
        exit()
    total += a[j]
    j += 1

print(total)