n = int(input())
a = list(map(int, input().split()))
count = 0
while True:
    positives = [x for x in a if x > 0]
    if len(positives) < 2:
        break
    a.sort(reverse=True)
    a[0] -= 1
    a[1] -= 1
    count += 1
print(count)