import bisect

n, m, d = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

b.sort()

max_sum = -1

for ai in a:
    lower = ai - d
    upper = ai + d
    idx = bisect.bisect_right(b, upper) - 1
    if idx >= 0 and b[idx] >= lower:
        current = ai + b[idx]
        if current > max_sum:
            max_sum = current

print(max_sum if max_sum != -1 else -1)