import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))
sum_a = sum(a)
if sum_a <= m:
    print("infinite")
else:
    a.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + a[i]
    low = 0
    high = a[-1]
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        pos = bisect.bisect_right(a, mid)
        total = prefix[pos] + mid * (n - pos)
        if total <= m:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)