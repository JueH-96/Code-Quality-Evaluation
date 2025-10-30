# YOUR CODE HERE
m, d = map(int, input().split())
y, mon, day = map(int, input().split())

day += 1
if day > d:
    day = 1
    mon += 1
    if mon > m:
        mon = 1
        y += 1

print(y, mon, day)