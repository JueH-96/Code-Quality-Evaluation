N, K = map(int, input().split())
A = list(map(int, input().split()))

count = 0
empty_seats = K
for i in range(N):
    if empty_seats < A[i]:
        count += 1
        empty_seats = K - A[i]
    else:
        empty_seats -= A[i]

if empty_seats > 0:
    count += 1

print(count)