import bisect

n = int(input())
a = list(map(int, input().split()))

count = 0
for i in range(n):
    # For mochi at position i as the top, find the first position j where a[j] >= 2 * a[i]
    # This means a[i] <= a[j] / 2, so mochi i can be placed on top of mochi j
    j = bisect.bisect_left(a, 2 * a[i])
    # All mochi from position j to n-1 can serve as the bottom
    count += n - j

print(count)