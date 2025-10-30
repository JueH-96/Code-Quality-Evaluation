import bisect

n = int(input())
A = list(map(int, input().split()))
A.sort()

total = 0
start = 0
while start < n:
    x_val = A[start]
    end = bisect.bisect_right(A, x_val) - 1
    m = end - start + 1
    Y = A[start:]
    sum_S = 0
    if Y:
        max_Y = Y[-1]
        k = 1
        while k * x_val <= max_Y:
            low = k * x_val
            high = (k + 1) * x_val - 1
            l = bisect.bisect_left(Y, low)
            r = bisect.bisect_right(Y, high)
            count = r - l
            sum_S += k * count
            k += 1
    contribution = m * (sum_S - m) + m * (m - 1) // 2
    total += contribution
    start = end + 1

print(total)