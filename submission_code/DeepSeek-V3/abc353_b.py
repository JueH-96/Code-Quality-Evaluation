# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))
empty_seats = K
count = 0
for a in A:
    if empty_seats < a:
        count += 1
        empty_seats = K
    empty_seats -= a
count += 1
print(count)