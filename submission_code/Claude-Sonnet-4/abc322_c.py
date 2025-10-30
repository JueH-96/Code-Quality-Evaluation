import bisect

N, M = map(int, input().split())
A = list(map(int, input().split()))

for i in range(1, N + 1):
    # Find the first fireworks day >= i
    idx = bisect.bisect_left(A, i)
    fireworks_day = A[idx]
    days_later = fireworks_day - i
    print(days_later)