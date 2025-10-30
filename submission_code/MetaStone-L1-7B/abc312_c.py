import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a_sorted = sorted(a)
b_sorted = sorted(b)

max_a = a_sorted[-1] if n else 0
max_b = b_sorted[-1] if m else 0

low = 0
high = max(max_a, max_b) + 1

def is_possible(x):
    count_a = bisect.bisect_right(a_sorted, x)
    count_b = len(b_sorted) - bisect.bisect_left(b_sorted, x)
    return count_a >= count_b

while low < high:
    mid = (low + high) // 2
    if is_possible(mid):
        high = mid
    else:
        low = mid + 1

print(low)