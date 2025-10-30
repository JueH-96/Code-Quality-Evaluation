n, k = map(int, input().split())
a = list(map(int, input().split()))
a.sort()

total = 0
left = 0
right = k - 1

while left < right:
    total += a[right] - a[right - 1]
    right -= 2
    if left + 1 <= right:
        total += a[left + 1] - a[left]
        left += 2
    else:
        break

print(total)