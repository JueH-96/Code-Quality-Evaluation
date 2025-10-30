import bisect

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    a.sort()

    max_gifts = 0

    for i in range(n):
        # Interval starting at a[i]
        start = a[i]
        end = start + m
        left_index = bisect.bisect_left(a, start)
        right_index = bisect.bisect_left(a, end)
        max_gifts = max(max_gifts, right_index - left_index)

        # Interval ending at a[i]
        end = a[i]
        start = end - m
        left_index = bisect.bisect_left(a, start)
        right_index = bisect.bisect_left(a, end)
        max_gifts = max(max_gifts, right_index - left_index)

    print(max_gifts)

solve()