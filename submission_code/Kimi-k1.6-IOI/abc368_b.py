n = int(input())
a = list(map(int, input().split()))
count = 0

while True:
    a.sort(reverse=True)
    if a[0] > 0 and a[1] > 0:
        a[0] -= 1
        a[1] -= 1
        count += 1
    else:
        break

print(count)