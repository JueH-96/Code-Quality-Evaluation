n = int(input())
a = list(map(int, input().split()))
count = 0

while True:
    a = sorted(a, reverse=True)
    if len(a) >= 2:
        a[0] -= 1
        a[1] -= 1
    count += 1
    positive = sum(1 for x in a if x > 0)
    if positive <= 1:
        break

print(count)