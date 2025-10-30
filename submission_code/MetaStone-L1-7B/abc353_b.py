n, k = map(int, input().split())
a = list(map(int, input().split()))
empty_seats = k
count = 0

while a:
    group = a.pop(0)
    if empty_seats < group:
        count += 1
        empty_seats = k
    else:
        empty_seats -= group

print(count)