# YOUR CODE HERE
n, k = map(int, input().split())
a = list(map(int, input().split()))

empty_seats = k
count = 0
i = 0
while i < n:
    if empty_seats < a[i]:
        count += 1
        empty_seats = k
    else:
        empty_seats -= a[i]
        i += 1
if i < n:
    count += 1
print(count)