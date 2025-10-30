import bisect

n, q = map(int, input().split())
a = list(map(int, input().split()))
a.sort()

for _ in range(q):
    b, k = map(int, input().split())
    pos = bisect.bisect_left(a, b)
    low = 0
    high = 10**18
    while low < high:
        mid = (low + high) // 2
        left_target = b - mid
        left_idx = bisect.bisect_left(a, left_target, 0, pos)
        left_count = pos - left_idx
        right_target = b + mid
        right_idx = bisect.bisect_right(a, right_target, pos, len(a))
        right_count = right_idx - pos
        total = left_count + right_count
        if total >= k:
            high = mid
        else:
            low = mid + 1
    print(low)