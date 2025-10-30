n, c = map(int, input().split())
times = list(map(int, input().split()))

if n == 0:
    print(0)
else:
    count = 1
    last = times[0]
    for t in times[1:]:
        if t - last >= c:
            count += 1
            last = t
    print(count)