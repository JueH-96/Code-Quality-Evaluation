# YOUR CODE HERE
n, k = map(int, input().split())
groups = list(map(int, input().split()))
started = 0
empty_seats = k

for group in groups:
    if empty_seats < group:
        started += 1
        empty_seats = k
    else:
        empty_seats -= group
        started += 1

print(started)