# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

starts = 0
empty_seats = K

for group in A:
    if group > empty_seats:
        starts += 1
        empty_seats = K
    empty_seats -= group

if empty_seats < K:
    starts += 1

print(starts)