import bisect

n, m, d = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
B.sort()

max_sum = -1

for a in A:
    high = a + d
    j = bisect.bisect_right(B, high) - 1
    if j >= 0 and B[j] >= a - d:
        current_sum = a + B[j]
        if current_sum > max_sum:
            max_sum = current_sum

print(max_sum if max_sum != -1 else -1)