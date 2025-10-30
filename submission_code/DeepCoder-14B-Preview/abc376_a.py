n, c = map(int, input().split())
times = list(map(int, input().split()))

count = 0
last = -float('inf')

for t in times:
    if last == -float('inf'):
        count += 1
        last = t
    else:
        if t - last >= c:
            count += 1
            last = t

print(count)