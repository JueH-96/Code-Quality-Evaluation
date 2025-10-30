import bisect

n, m, d = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
A.sort()
B.sort()
max_sum = -1

for a in A:
    low = a - d
    high = a + d
    idx = bisect.bisect_right(B, high)
    if idx > 0:
        candidate = B[idx-1]
        if candidate >= low:
            current_sum = a + candidate
            if current_sum > max_sum:
                max_sum = current_sum

print(max_sum if max_sum != -1 else -1)