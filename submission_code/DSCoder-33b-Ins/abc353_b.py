# YOUR CODE HERE

N, K = map(int, input().split())
A = list(map(int, input().split()))

attractions_started = 0
empty_seats = K

for i in range(N):
    if empty_seats < A[i]:
        attractions_started += 1
        empty_seats = K
    else:
        empty_seats -= A[i]

print(attractions_started)