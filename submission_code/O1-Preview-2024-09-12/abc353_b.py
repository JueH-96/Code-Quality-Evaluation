# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))
empty_seats = K
attr_started = 0
i = 0
while i < N:
    if empty_seats < A[i]:
        attr_started += 1
        empty_seats = K
    else:
        empty_seats -= A[i]
        i += 1
attr_started += 1
print(attr_started)