N, K = map(int, input().split())
A = list(map(int, input().split()))

empty_seats = K
starts = 0
i = 0

while i < N:
    if A[i] > empty_seats:
        starts += 1
        empty_seats = K
    else:
        empty_seats -= A[i]
        i += 1

if i == N and empty_seats < K:
    starts += 1

print(starts)