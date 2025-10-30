n = int(input())
a = list(map(int, input().split()))
count = 0

while True:
    a.sort(reverse=True)
    positive_count = sum(1 for x in a if x > 0)
    if positive_count < 2:
        break
    a[0] -= 1
    a[1] -= 1
    count += 1

print(count)