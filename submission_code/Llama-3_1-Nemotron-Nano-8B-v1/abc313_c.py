n, *rest = map(int, open(0).read().split())
a = rest[:n]
a.sort()

total = sum(a)
if total % n == 0:
    target = total // n
    res = sum(x - target for x in a if x > target)
else:
    m = total // n
    r = total % n
    target = [m] * (n - r) + [m + 1] * r
    target.sort()
    total_diff = sum(abs(x - y) for x, y in zip(a, target))
    res = total_diff // 2

print(res)