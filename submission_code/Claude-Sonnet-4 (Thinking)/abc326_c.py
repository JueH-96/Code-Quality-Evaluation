n, m = map(int, input().split())
a = list(map(int, input().split()))

a.sort()

max_gifts = 0
left = 0

for right in range(n):
    while a[right] - a[left] >= m:
        left += 1
    max_gifts = max(max_gifts, right - left + 1)

print(max_gifts)