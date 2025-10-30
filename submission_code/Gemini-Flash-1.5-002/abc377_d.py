n, m = map(int, input().split())
intervals = []
for i in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

count = 0
for l in range(1, m + 1):
    for r in range(l, m + 1):
        valid = True
        for interval in intervals:
            if interval[0] >= l and interval[1] <= r:
                valid = False
                break
        if valid:
            count += 1

print(count)